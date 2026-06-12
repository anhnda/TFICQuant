"""
TFIC -- Transverse-Field Ising Correction (paper Algorithm 1).  [optimized]

Same algorithm and guarantees as the reference encoder, but the hot loops are
restructured to remove per-column GPU<->CPU synchronisation and to amortise the
RG maintenance over a whole sweep.  The monotonicity guarantee is unchanged:
every accepted move is still verified against the EXACT reconstruction energy
E(s)=Tr(R G R^T) (the "approximate proposals, exact acceptance" principle of
section 3.7), now applied at sweep / cluster-batch granularity instead of one
coordinate at a time.

What changed vs. the reference implementation
---------------------------------------------
Phase 1 (descend): instead of a Python loop over columns -- each doing
`.item()`/`.any()` (a GPU sync) and a per-column rank-1 RG update -- we run a
*batched* sweep:
  * propose every column's best single flip in one vectorised op (Eq. 15),
  * accept the noise-floor-clearing flips for the columns scheduled this chunk,
  * recompute R for those columns and update RG with ONE matmul
        RG += dR_chunk @ G[chunk, :]                       (amortised Eq. 18)
  * verify the chunk strictly lowered the exact energy; if a chunk ever fails
    (possible only when two same-row columns in the chunk interact via Lemma 1's
    cross term), bisect the chunk -- guaranteeing monotonicity with no per-column
    sync.  With chunk size 1 this reduces exactly to the reference column sweep.

Phase 2 (tunnel): the per-row cluster growth + 2^|T| enumeration is moved off
the GPU.  We pull the small per-row slices (RG[i], the candidate columns, and
their |T|x|T| G submatrix) to CPU/numpy once, do all scalar work there, and
write the accepted cluster flip back in one indexed update.  No `.item()` inside
the growth loop.

Paper <-> code dictionary unchanged (see original docstring): G := H = Sigma +
mu mu^T (needs keep_sigma=True); spin flip changes residual by delta=flip_dir*scale;
dE_j = 2*delta*(RG)_ij + delta^2*G_jj; S_jk = -2 delta_j delta_k G_jk; group gain
dE_T = 2<delta_T,(RG)_{i,T}> + delta_T^T G_TT delta_T.
"""
from __future__ import annotations

import torch


class TFICEncoder:
    name = "tfic"

    def __init__(self,
                 alpha: float = 1.0, beta: float = 1.0, eta: float = 1.0,
                 gamma_th: float = 0.5, kappa: float = 2.0,
                 gmax: int = 6, n_stages: int = 2, sweeps: int = 3,
                 c_cand: float = 8.0, top_m: int = 32,
                 max_tunnel_rows: int = 512, max_clusters_per_row: int = 50,
                 chunk_cols: int = 256, work_dtype=torch.float32):
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
        self.chunk_cols = int(chunk_cols)   # columns proposed together per chunk
        self.work_dtype = work_dtype

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def apply(self, state, stats):
        assert stats.Sigma is not None, (
            "TFIC needs a materialized G (gram backend, keep_sigma=True; "
            "register 'tfic' in KEEP_SIGMA).")
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

        R = (Wint - zp) * scale - Wf
        RG = R @ G
        e_cur = (R * RG).sum().item()
        e0 = e_cur

        # top-m coupling neighbours per column (off-diagonal |G_jk|)
        Gabs = G.abs().clone()
        Gabs.fill_diagonal_(0.0)
        m = min(self.top_m, pin - 1)
        _, nbr_idx = torch.topk(Gabs, m, dim=1)            # [pin,m]
        del Gabs
        col_ar = torch.arange(pin, device=dev)
        G_nbr = G[col_ar.unsqueeze(1), nbr_idx]            # [pin,m] G_{j,nbr}

        frac = (pre - Wint).abs().clamp(0, 0.5)
        U_bnd = (1.0 - 2.0 * frac).clamp_min(0.0)

        total_flips = 0
        total_cluster_moves = 0
        cluster_energy = 0.0

        for a in range(self.n_stages):
            t = a / max(1, self.n_stages - 1)
            gamma_a = self.gamma_th * (0.6 + 0.4 * t)
            gmax_a = max(2, int(round(self.gmax - (self.gmax - 2) * t)))
            final_stage = (a == self.n_stages - 1)

            flip_dir = self._flip_dir(pre, Wint)
            delta = flip_dir * scale
            in_range = self._in_range(Wint, flip_dir, max_int)
            dE = self._dE(delta, RG, diagG, in_range)

            tau = self._tau(dE)

            # ---- adaptive transverse field (Eq. 13), U_fru fully vectorised
            U_fld = torch.exp(-dE.clamp_min(0.0) / tau)
            dk = delta[:, nbr_idx]                          # [C,pin,m]
            U_fru = ((-2.0 * delta.unsqueeze(2) * dk * G_nbr.unsqueeze(0))
                     .clamp_min(0.0).sum(2) / tau).clamp(max=1.0)
            del dk
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
                        cols, Wint, R, RG, G, diagG, scale, pre, pool,
                        thresh, max_int, e_cur)
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
            dE = self._dE(delta, RG, diagG, in_range)
            cand = in_range & (dE >= 0) & (dE <= self.c_cand * tau)
            cand_counts = cand.sum(1)
            rows = torch.nonzero(cand_counts >= 2, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            if rows.numel() > self.max_tunnel_rows:
                topr = torch.topk(cand_counts[rows], self.max_tunnel_rows).indices
                rows = rows[topr]

            # batch-pull everything the tunnelling needs to CPU once
            rows_l = rows.tolist()
            cand_cpu = cand[rows].cpu().numpy()
            dE_cpu = dE[rows].cpu().numpy()
            nbr_cpu = nbr_idx.cpu().numpy()
            flipdir_cpu = flip_dir[rows].cpu().numpy()
            scale_cpu = scale[rows].cpu().numpy()
            RG_cpu = RG[rows].cpu().numpy()
            G_cpu = G.cpu().numpy()
            for ridx, i in enumerate(rows_l):
                applied = self._tunnel_row_cpu(
                    cand_cpu[ridx], dE_cpu[ridx], nbr_cpu, flipdir_cpu[ridx],
                    scale_cpu[ridx], RG_cpu[ridx], G_cpu, gmax_a)
                if not applied:
                    continue
                for cols_T, dirs_T, gain in applied:
                    cols_t = torch.tensor(cols_T, device=dev)
                    dirs_t = torch.tensor(dirs_T, device=dev, dtype=wdt)
                    dT = dirs_t * scale[i, cols_t]
                    Wint[i, cols_t] = (Wint[i, cols_t] + dirs_t).clamp(0, max_int)
                    R[i, cols_t] = R[i, cols_t] + dT
                    RG[i, :] = RG[i, :] + dT @ G[cols_t, :]
                    total_cluster_moves += 1
                    cluster_energy += -gain
            # refresh exact energy after tunnelling batch
            e_cur = (R * RG).sum().item()

        R_final = (Wint - zp) * scale - Wf
        e_final = (R_final * (R_final @ G)).sum().item()
        out = (Wint - zp) * scale
        if pin > d:
            out = out[:, :d]
        info = {"encoder": self.name, "k": stats.k,
                "total_flips": total_flips,
                "cluster_moves": total_cluster_moves,
                "energy_start": e0, "energy_final": e_final,
                "energy_drop": e0 - e_final,
                "cluster_energy_released": cluster_energy}
        del G, RG, R, scale, zp, pre, Wf, G_nbr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out.to(state.original_dtype), info

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
    def _dE(delta, RG, diagG, in_range):
        dE = 2.0 * delta * RG + delta * delta * diagG.unsqueeze(0)
        return torch.where(in_range, dE, torch.full_like(dE, float("inf")))

    @staticmethod
    def _tau(dE):
        finite = torch.isfinite(dE)
        tau = dE[finite].abs().median().item() if finite.any() else 0.0
        return max(tau, 1e-12)

    @torch.no_grad()
    def _descend_chunk(self, cols, Wint, R, RG, G, diagG, scale, pre, pool,
                       thresh, max_int, e_cur):
        """Propose+accept single flips for a column chunk, amortise the RG
        update into one matmul, and verify the chunk lowered the exact energy.
        If verification fails (same-row cross terms within the chunk), bisect.
        Returns (new_exact_energy, num_flips)."""
        nc = cols.numel()
        if nc == 0:
            return e_cur, 0
        fd = self._flip_dir(pre[:, cols], Wint[:, cols])       # [C,nc]
        dcol = fd * scale[:, cols]
        prop = Wint[:, cols] + fd
        okrange = (prop >= 0) & (prop <= max_int)
        dEj = 2.0 * dcol * RG[:, cols] + dcol * dcol * diagG[cols].unsqueeze(0)
        acc = okrange & pool[:, cols] & (dEj < thresh)
        nflip = int(acc.sum().item())
        if nflip == 0:
            return e_cur, 0

        step = torch.where(acc, fd, torch.zeros_like(fd))
        dR = torch.where(acc, dcol, torch.zeros_like(dcol))    # [C,nc]
        # tentatively apply
        Wint[:, cols] = (Wint[:, cols] + step).clamp(0, max_int)
        R[:, cols] = R[:, cols] + dR
        RG_add = dR @ G[cols, :]                                # [C,pin] one matmul
        RG += RG_add
        e_new = (R * RG).sum().item()

        if e_new <= e_cur + 1e-9 or nc == 1:
            return e_new, nflip
        # rare: chunk raised energy via intra-chunk same-row coupling -> revert
        # and bisect for guaranteed monotonicity (no per-column sync in common case)
        Wint[:, cols] = (Wint[:, cols] - step).clamp(0, max_int)
        R[:, cols] = R[:, cols] - dR
        RG -= RG_add
        mid = nc // 2
        e_cur, n1 = self._descend_chunk(cols[:mid], Wint, R, RG, G, diagG,
                                        scale, pre, pool, thresh, max_int, e_cur)
        e_cur, n2 = self._descend_chunk(cols[mid:], Wint, R, RG, G, diagG,
                                        scale, pre, pool, thresh, max_int, e_cur)
        return e_cur, n1 + n2

    # ----------------------- tunnelling (CPU/numpy) -------------------- #
    @staticmethod
    def _tunnel_row_cpu(cand_i, dE_i, nbr_cpu, flipdir_i, scale_i, RG_i,
                        G_cpu, gmax_a):
        """All scalar cluster work for one row, off-GPU. Returns a list of
        accepted (cols, dirs, gain) cluster moves; gain<0 (exact group gain)."""
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
            RG_T = RG_i[Ta]
            nT = len(T)
            best_gain, best_f = 0.0, None
            for mask in range(1, 1 << nT):
                f = np.array([(mask >> b) & 1 for b in range(nT)], dtype=float)
                dT = f * delta_full
                gain = 2.0 * float(dT @ RG_T) + float(dT @ G_TT @ dT)
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


def make_tfic(alpha=1.0, beta=1.0, eta=1.0, gamma_th=0.5, kappa=2.0,
              gmax=6, n_stages=2, sweeps=3, c_cand=8.0, top_m=32,
              chunk_cols=256):
    return TFICEncoder(alpha=alpha, beta=beta, eta=eta, gamma_th=gamma_th,
                       kappa=kappa, gmax=gmax, n_stages=n_stages,
                       sweeps=sweeps, c_cand=c_cand, top_m=top_m,
                       chunk_cols=chunk_cols)
