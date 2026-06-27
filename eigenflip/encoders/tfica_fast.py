"""
TFIC-A -- Transverse-Field Ising Correction with ASYMMETRIC calibration.
[optimized; derived from tfic_fast.py]

This is the asymmetric-calibration extension of TFIC (paper subsection
"Asymmetric-Calibration Field (TFIC-A)"). It imports GPTAQ's insight that the
correct reconstruction target is the layer output on the *full-precision* input
X~ that upstream layers would have emitted, not the quantized input X the layer
actually receives. Writing dY = W(X~ - X) for the per-token output deviation,
the asymmetric objective is

    E_asym(s) = (1/n) || X_sc R(s)^T - dY^T ||_F^2.

Theorem (coupling invariance, proven in the paper): expanding E_asym leaves the
Ising COUPLINGS J_jk = H_ij H_ik G_jk identical to the symmetric case; the only
change is an additive shift to the external field via the single statistic

    F = (1/n) X_sc^T dY^T          (shape [d_in, C]).

Concretely every flip gain picks up exactly one extra term:

    dE_j      = 2 delta_j [ (RG)_ij - F^T_ij ] + delta_j^2 G_jj      (was: -0)
    dE_T      = 2 <delta_T, (RG)_{i,T} - F^T_{i,T}> + delta_T^T G_TT delta_T

with F^T in [C, pin] sharing RG's layout. EVERYTHING ELSE -- pair-flip identity,
synergy score S_jk = -2 delta_j delta_k G_jk (field-INDEPENDENT), barrier /
frustration analysis, certified fixing dominance bound (coupling-only), the
rank-1 RG maintenance, noise floor, and the exact-acceptance monotonicity
guarantee -- transfers verbatim. F is a fixed per-layer offset (does not depend
on s), so RG maintenance is untouched; we just fold (RG - RGF) wherever the
symmetric code used RG, where RGF := F^T is precomputed once.

If stats.F is None the encoder is bit-identical to symmetric tfic_fast: the
shift collapses to zero and E_asym == E_rec. This is the clean fallback that
also serves as the field-only ablation control.

NOTE (noise caveat, from the paper): dY is a difference of two nearly-equal
large quantities (W X~ vs W X), so F has a much lower SNR than G, and the gap
widens with depth. The noise floor (kappa*tau) prunes shifts dominated by
estimation noise, so on deep layers TFIC-A is expected to degrade toward
symmetric TFIC. The info dict reports the field's contribution so the
crossover can be measured (decisive ablation T-asym).
"""
from __future__ import annotations

import torch


class TFICAEncoder:
    name = "tfica"

    def __init__(self,
                 alpha: float = 1.0, beta: float = 1.0, eta: float = 1.0,
                 gamma_th: float = 0.5, kappa: float = 2.0,
                 gmax: int = 6, n_stages: int = 2, sweeps: int = 3,
                 c_cand: float = 8.0, top_m: int = 32,
                 max_tunnel_rows: int = 512, max_clusters_per_row: int = 50,
                 chunk_cols: int = 256, work_dtype=torch.float32,
                 field_scale: float = 0.25, guard_ratio: float = 5.0):
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.gamma_th = gamma_th
        self.kappa = kappa
        self.gmax = int(gmax)
        self.n_stages = int(n_stages)
        self.sweeps = int(sweeps)
        self.c_cand = c_cand
        self.top_m = int(top_m)
        self.max_tunnel_rows = int(max_tunnel_rows)
        self.max_clusters_per_row = int(max_clusters_per_row)
        self.chunk_cols = int(chunk_cols)
        self.work_dtype = work_dtype
        self.field_scale = float(field_scale)   # lambda_F: damp the asym field
        self.guard_ratio = float(guard_ratio)   # max sym-energy blow-up allowed

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def apply(self, state, stats):
        assert stats.Sigma is not None, (
            "TFIC-A needs a materialized G (gram backend, keep_sigma=True; "
            "register 'tfica_fast' in KEEP_SIGMA).")
        dev = state.scale.device
        wdt = self.work_dtype
        d = stats.d
        pin = state.padded_in_features

        # ---- exact reconstruction metric G = H = Sigma + mu mu^T
        mu = stats.mu_hat.to(device=dev, dtype=wdt)
        G = stats.Sigma.to(device=dev, dtype=wdt) + torch.outer(mu, mu)
        if pin > d:
            Gp = torch.zeros(pin, pin, device=dev, dtype=wdt)
            Gp[:d, :d] = G
            idx = torch.arange(d, pin, device=dev)
            Gp[idx, idx] = torch.diagonal(G).mean()
            G = Gp
        diagG = torch.diagonal(G).contiguous()

        scale = state.scale.to(wdt)
        zp = state.zero_point.to(wdt)
        pre = state.pre_round.to(wdt)
        Wf = state.float_weights.to(wdt)
        Wint = state.integer_weights.to(wdt).clone()
        C = Wint.shape[0]
        max_int = float(state.max_int)

        # ---- asymmetric field shift.  stats.F now holds the INPUT-SPACE
        # cross-Gram K = (1/n) ΔA Aᵀ  ([d_in, d_in], = GPTAQ dXXT/n), NOT a
        # per-row field. The objective is  E_asym = Tr[R G Rᵀ] - 2 Tr[R K Wᵀ],
        # so the driving-field shift is  W K  (shape [C, pin], same layout as RG):
        #   ∂E_asym/∂R_ij = 2(RG)_ij - 2 (W K)_ij.
        # field_scale = alpha (GPTAQ uses 0.25) damps the correction.
        if getattr(stats, "F", None) is not None:
            K = stats.F.to(device=dev, dtype=wdt)            # [d_in, d_in]
            d_in = K.shape[0]
            # Wf is [C, d] (unpadded); pad columns to pin to match K and RG.
            if Wf.shape[1] < pin:
                Wfull = torch.zeros(C, pin, device=dev, dtype=wdt)
                Wfull[:, :Wf.shape[1]] = Wf
            else:
                Wfull = Wf
            if d_in < pin:
                Kp = torch.zeros(pin, pin, device=dev, dtype=wdt)
                Kp[:d_in, :d_in] = K
                K = Kp
            RGF = (Wfull @ K.t()).contiguous()               # [C, pin] = W Kᵀ
            #   ∂/∂R_ij Tr[R K Wᵀ] = (W Kᵀ)_ij  (K is NOT symmetric)
            if self.field_scale != 1.0:
                RGF = RGF * self.field_scale
            del K
            asym = True
        else:
            RGF = torch.zeros(C, pin, device=dev, dtype=wdt)
            asym = False

        R = (Wint - zp) * scale - Wf
        RG = R @ G
        # effective driving field used by every flip gain: (RG - F^T)
        RGe = RG - RGF
        e_cur = (R * RG).sum().item()    # exact symmetric energy term (report)
        e0 = e_cur

        # top-m coupling neighbours per column (off-diagonal |G_jk|)
        Gabs = G.abs().clone()
        Gabs.fill_diagonal_(0.0)
        m = min(self.top_m, pin - 1)
        _, nbr_idx = torch.topk(Gabs, m, dim=1)            # [pin,m]
        del Gabs
        col_ar = torch.arange(pin, device=dev)
        G_nbr = G[col_ar.unsqueeze(1), nbr_idx]            # [pin,m]

        frac = (pre - Wint).abs().clamp(0, 0.5)
        U_bnd = (1.0 - 2.0 * frac).clamp_min(0.0)

        total_flips = 0
        total_cluster_moves = 0
        cluster_energy = 0.0
        field_flips = 0   # flips whose acceptance flips sign vs symmetric dE

        for a in range(self.n_stages):
            t = a / max(1, self.n_stages - 1)
            gamma_a = self.gamma_th * (0.6 + 0.4 * t)
            gmax_a = max(2, int(round(self.gmax - (self.gmax - 2) * t)))
            final_stage = (a == self.n_stages - 1)

            flip_dir = self._flip_dir(pre, Wint)
            delta = flip_dir * scale
            in_range = self._in_range(Wint, flip_dir, max_int)
            dE = self._dE(delta, RGe, diagG, in_range)
            if asym:
                dE_sym = self._dE(delta, RG, diagG, in_range)
                field_flips += int(((dE < 0) ^ (dE_sym < 0)).sum().item())
                del dE_sym

            tau = self._tau(dE)

            # ---- adaptive transverse field (Eq. 13)
            U_fld = torch.exp(-dE.clamp_min(0.0) / tau)
            U_fru = torch.empty_like(delta)
            cs_fru = max(1, self.chunk_cols)
            for j0 in range(0, pin, cs_fru):
                j1 = min(j0 + cs_fru, pin)
                d_blk = delta[:, j0:j1]
                dk = delta[:, nbr_idx[j0:j1]]
                contrib = (-2.0 * d_blk.unsqueeze(2) * dk
                           * G_nbr[j0:j1].unsqueeze(0))
                U_fru[:, j0:j1] = (contrib.clamp_min(0.0).sum(2)
                                   / tau).clamp(max=1.0)
                del dk, contrib, d_blk
            Gamma = self.alpha * U_bnd + self.beta * U_fld + self.eta * U_fru
            Gamma = torch.where(scale > 0, Gamma, torch.zeros_like(Gamma))
            pool = (Gamma > gamma_a) & in_range
            thresh = -self.kappa * tau

            # ============= Phase 1: batched descent ==================== #
            for _ in range(self.sweeps):
                perm = torch.randperm(pin, device=dev)
                moved = 0
                cs = max(1, self.chunk_cols)
                for c0 in range(0, pin, cs):
                    cols = perm[c0:c0 + cs]
                    e_cur, nflip = self._descend_chunk(
                        cols, Wint, R, RG, RGF, RGe, G, diagG, scale, pre,
                        pool, thresh, max_int, e_cur)
                    moved += nflip
                total_flips += moved
                if moved == 0:
                    break

            # ================ Phase 2: tunnel ========================== #
            if final_stage or self.gmax < 2:
                continue
            flip_dir = self._flip_dir(pre, Wint)
            delta = flip_dir * scale
            in_range = self._in_range(Wint, flip_dir, max_int)
            dE = self._dE(delta, RGe, diagG, in_range)
            cand = in_range & (dE >= 0) & (dE <= self.c_cand * tau)
            cand_counts = cand.sum(1)
            rows = torch.nonzero(cand_counts >= 2, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            if rows.numel() > self.max_tunnel_rows:
                topr = torch.topk(cand_counts[rows], self.max_tunnel_rows).indices
                rows = rows[topr]

            rows_l = rows.tolist()
            cand_cpu = cand[rows].cpu().numpy()
            dE_cpu = dE[rows].cpu().numpy()
            nbr_cpu = nbr_idx.cpu().numpy()
            flipdir_cpu = flip_dir[rows].cpu().numpy()
            scale_cpu = scale[rows].cpu().numpy()
            RGe_cpu = RGe[rows].cpu().numpy()      # driving field for group gain
            G_cpu = G.cpu().numpy()
            for ridx, i in enumerate(rows_l):
                applied = self._tunnel_row_cpu(
                    cand_cpu[ridx], dE_cpu[ridx], nbr_cpu, flipdir_cpu[ridx],
                    scale_cpu[ridx], RGe_cpu[ridx], G_cpu, gmax_a)
                if not applied:
                    continue
                for cols_T, dirs_T, gain in applied:
                    cols_t = torch.tensor(cols_T, device=dev)
                    dirs_t = torch.tensor(dirs_T, device=dev, dtype=wdt)
                    dT = dirs_t * scale[i, cols_t]
                    Wint[i, cols_t] = (Wint[i, cols_t] + dirs_t).clamp(0, max_int)
                    R[i, cols_t] = R[i, cols_t] + dT
                    upd = dT @ G[cols_t, :]
                    RG[i, :] = RG[i, :] + upd
                    RGe[i, :] = RGe[i, :] + upd     # F is constant; only RG part moves
                    total_cluster_moves += 1
                    cluster_energy += -gain
            e_cur = (R * RG).sum().item()

        R_final = (Wint - zp) * scale - Wf
        e_final = (R_final * (R_final @ G)).sum().item()

        # SAFETY GUARD. The asymmetric field pulls R away from 0 to track the
        # upstream-error target dY. With a base like RTN at 3-bit the upstream
        # error -- and hence F -- can be large, and an over-strong pull can blow
        # up the layer output (cascading to nonsense PPL). TFIC's monotonicity is
        # on E_asym, NOT on the plain reconstruction Tr(R G R^T); so we cap how
        # far the *symmetric* energy is allowed to grow. If a layer's symmetric
        # reconstruction degrades past `guard_ratio` x the RTN baseline, we treat
        # the asymmetric correction as unsafe for that layer and fall back to the
        # symmetric-TFIC result (recomputed by ignoring F) -- never worse than
        # tfic_fast.
        if asym and self._guard_tripped(e0, e_final):
            print(f"    [asym] GUARD: sym energy {e0:.3e}->{e_final:.3e} "
                  f"exceeds {self.guard_ratio}x -> fallback to symmetric TFIC")
            return self._symmetric_fallback(state, stats)

        out = (Wint - zp) * scale
        if pin > d:
            out = out[:, :d]
        info = {"encoder": self.name, "k": stats.k,
                "asymmetric": asym,
                "total_flips": total_flips,
                "field_sign_flips": field_flips,   # flips the F-shift created/killed
                "cluster_moves": total_cluster_moves,
                "energy_start": e0, "energy_final": e_final,
                "energy_drop": e0 - e_final,
                "cluster_energy_released": cluster_energy}
        del G, RG, RGe, RGF, R, scale, zp, pre, Wf, G_nbr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out.to(state.original_dtype), info

    def _guard_tripped(self, e0, e_final):
        if not (e0 == e0 and e_final == e_final):   # NaN
            return True
        if e_final != e_final or e_final == float("inf"):
            return True
        base = abs(e0) + 1e-12
        return e_final > self.guard_ratio * base

    @torch.no_grad()
    def _symmetric_fallback(self, state, stats):
        """Re-run with the field disabled => identical to symmetric tfic_fast.
        Guarded against recursion by temporarily detaching stats.F."""
        savedF = getattr(stats, "F", None)
        try:
            stats.F = None
            out, info = self.apply(state, stats)
        finally:
            stats.F = savedF
        info["asymmetric"] = False
        info["asym_fallback"] = True
        return out, info

    # ------------------------- helpers (GPU) --------------------------- #
    @staticmethod
    def _flip_dir(pre, Wint):
        fd = torch.sign(pre - Wint)
        return torch.where(fd == 0, torch.ones_like(fd), fd)

    @staticmethod
    def _in_range(Wint, flip_dir, max_int):
        prop = Wint + flip_dir
        return (prop >= 0) & (prop <= max_int)

    @staticmethod
    def _dE(delta, RGe, diagG, in_range):
        # dE_j = 2 delta (RG - F^T)_ij + delta^2 G_jj  (RGe already = RG - F^T)
        dE = 2.0 * delta * RGe + delta * delta * diagG.unsqueeze(0)
        return torch.where(in_range, dE, torch.full_like(dE, float("inf")))

    @staticmethod
    def _tau(dE):
        finite = torch.isfinite(dE)
        tau = dE[finite].abs().median().item() if finite.any() else 0.0
        return max(tau, 1e-12)

    @torch.no_grad()
    def _descend_chunk(self, cols, Wint, R, RG, RGF, RGe, G, diagG, scale, pre,
                       pool, thresh, max_int, e_cur):
        """Propose+accept single flips for a column chunk, amortise the RG
        update into one matmul, verify the chunk lowered the EXACT asymmetric
        energy E_asym = Tr(R G R^T) - 2 Tr(R F).  RGe = RG - F^T is the driving
        field (kept in sync with RG since F is constant)."""
        nc = cols.numel()
        if nc == 0:
            return e_cur, 0
        fd = self._flip_dir(pre[:, cols], Wint[:, cols])
        dcol = fd * scale[:, cols]
        prop = Wint[:, cols] + fd
        okrange = (prop >= 0) & (prop <= max_int)
        dEj = (2.0 * dcol * RGe[:, cols]
               + dcol * dcol * diagG[cols].unsqueeze(0))
        acc = okrange & pool[:, cols] & (dEj < thresh)
        nflip = int(acc.sum().item())
        if nflip == 0:
            return e_cur, 0

        step = torch.where(acc, fd, torch.zeros_like(fd))
        dR = torch.where(acc, dcol, torch.zeros_like(dcol))
        Wint[:, cols] = (Wint[:, cols] + step).clamp(0, max_int)
        R[:, cols] = R[:, cols] + dR
        RG_add = dR @ G[cols, :]
        RG += RG_add
        RGe += RG_add                       # F constant => RGe tracks RG exactly
        # exact asymmetric energy: symmetric quadratic minus 2*Tr(R F)
        #   Tr(R F) = sum_ij R_ij F^T_ij = (R * RGF).sum()
        e_new = (R * RG).sum().item() - 2.0 * (R * RGF).sum().item()

        if e_new <= e_cur + 1e-9 or nc == 1:
            return e_new, nflip
        # rare: intra-chunk same-row cross term raised energy -> revert + bisect
        Wint[:, cols] = (Wint[:, cols] - step).clamp(0, max_int)
        R[:, cols] = R[:, cols] - dR
        RG -= RG_add
        RGe -= RG_add
        mid = nc // 2
        e_cur, n1 = self._descend_chunk(cols[:mid], Wint, R, RG, RGF, RGe, G,
                                        diagG, scale, pre, pool, thresh,
                                        max_int, e_cur)
        e_cur, n2 = self._descend_chunk(cols[mid:], Wint, R, RG, RGF, RGe, G,
                                        diagG, scale, pre, pool, thresh,
                                        max_int, e_cur)
        return e_cur, n1 + n2

    # ----------------------- tunnelling (CPU/numpy) -------------------- #
    @staticmethod
    def _tunnel_row_cpu(cand_i, dE_i, nbr_cpu, flipdir_i, scale_i, RGe_i,
                        G_cpu, gmax_a):
        """Per-row cluster work, off-GPU.  Synergy S_jk = -2 delta_j delta_k
        G_jk is field-INDEPENDENT (unchanged from symmetric).  Only the group
        gain's linear part uses the driving field RGe = RG - F^T:
            dE_T = 2 <delta_T, RGe_T> + delta_T^T G_TT delta_T."""
        import numpy as np
        cand_cols = np.nonzero(cand_i)[0]
        if cand_cols.size < 2:
            return []
        cset = set(int(c) for c in cand_cols)
        ranked = cand_cols[np.argsort(dE_i[cand_cols])]
        used = set()
        out = []
        clusters = 0
        for seed in ranked:
            seed = int(seed)
            if clusters >= 50 or seed in used:
                continue
            T = [seed]
            while len(T) < gmax_a:
                best_k, best_syn = None, 0.0
                for member in T:
                    for k in nbr_cpu[member]:
                        k = int(k)
                        if k in T or k in used or k not in cset:
                            continue
                        dk = flipdir_i[k] * scale_i[k]
                        syn = 0.0
                        for mm in T:
                            dm = flipdir_i[mm] * scale_i[mm]
                            syn += -2.0 * dm * dk * G_cpu[mm, k]
                        if syn > best_syn:
                            best_syn, best_k = syn, k
                if best_k is None or best_syn <= 0.0:
                    break
                T.append(best_k)
            clusters += 1
            if len(T) < 2:
                continue
            Ta = np.array(T)
            flip_T = flipdir_i[Ta]
            scale_T = scale_i[Ta]
            delta_full = flip_T * scale_T
            G_TT = G_cpu[np.ix_(Ta, Ta)]
            RGe_T = RGe_i[Ta]                 # driving field (RG - F^T)
            nT = len(T)
            best_gain, best_f = 0.0, None
            for mask in range(1, 1 << nT):
                f = np.array([(mask >> b) & 1 for b in range(nT)], dtype=float)
                dT = f * delta_full
                gain = 2.0 * float(dT @ RGe_T) + float(dT @ G_TT @ dT)
                if gain < best_gain:
                    best_gain, best_f = gain, f
            if best_f is None or best_gain >= 0.0:
                continue
            sel = best_f.astype(bool)
            cols_T = [int(c) for c in Ta[sel]]
            dirs_T = [float(flip_T[idx]) for idx in range(nT) if sel[idx]]
            out.append((cols_T, dirs_T, best_gain))
            used.update(T)
        return out


def make_tfica(alpha=1.0, beta=1.0, eta=1.0, gamma_th=0.5, kappa=2.0,
               gmax=6, n_stages=2, sweeps=3, c_cand=8.0, top_m=32,
               chunk_cols=256):
    return TFICAEncoder(alpha=alpha, beta=beta, eta=eta, gamma_th=gamma_th,
                        kappa=kappa, gmax=gmax, n_stages=n_stages,
                        sweeps=sweeps, c_cand=c_cand, top_m=top_m,
                        chunk_cols=chunk_cols)
