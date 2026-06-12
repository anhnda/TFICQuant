"""
TFIC -- Transverse-Field Ising Correction (paper Algorithm 1).

A barrier-aware discrete corrector that re-optimizes the rounding decisions of
any base (RTN/AWQ) state. It is a *post-hoc* encoder in the exact same contract
as DenseGPTQ: consume an IntegerQuantizedTensorState + LayerStats, return
corrected dequantized weights and a diagnostics dict.

Paper <-> code dictionary
-------------------------
* Paper reconstruction metric  G = X_sc^T X_sc / n   (the layer Hessian)
    == this code's second moment  H = Sigma + mu mu^T.
  So TFIC needs `stats.Sigma` materialized (gram backend, keep_sigma=True),
  exactly like the `gptq` encoder. We build  G := H  once per layer.
* A rounding bit is an Ising spin s_ij in {-1,+1}; flipping code (i,j) by
  flip_dir changes the dequantized weight by  delta = flip_dir * scale_ij,
  and changes the residual  R = Wq - W  at (i,j) by  +delta.
* Single-flip gain (Eq. 15):   dE_j = 2*delta*(R G)_ij + delta^2 * G_jj
* Pair-flip gain (Lemma 1):     dE_{j,k} = dE_j + dE_k + 2*delta_j*delta_k*G_jk
* Barrier synergy (Eq. 12):     S_jk = -2*delta_j*delta_k*G_jk
* Group-flip gain (Eq. 16):     dE_T = 2<delta_T,(R G)_{i,T}> + delta_T^T G_TT delta_T

Everything is measured against the EXACT reconstruction energy
    E(s) = Tr( R G R^T ),   R = Wq(s) - W
so every accepted move is verified exactly (the "approximate proposals, exact
acceptance" principle of section 3.7).  Monotone, finite-converging descent;
synergy-clustered tunnelling crosses the barriers that single-flip descent
cannot.

Knobs (paper notation):
  alpha,beta,eta : transverse-field weights (1,0,0) = boundary-only (section 4 prelim)
  gamma_th       : active-pool threshold on the field
  kappa          : noise-floor multiplier (Eq. 20); kappa=0 -> unguarded
  gmax           : max cluster size for tunnelling (<=6)
  n_stages       : annealing stages A (final stage is pure CD, no tunnel)
  sweeps         : CD sweeps per stage T
  c_cand         : candidate band 0<=dE<=c_cand*tau for tunnelling seeds
  top_m          : neighbours per spin used for synergy / frustration
  max_tunnel_rows: row cap for tunnelling (paper caps N_rows<=512)
"""
from __future__ import annotations

import torch

from ..quantization.state import IntegerQuantizedTensorState
from ..statistics.trust_region import LayerStats


class TFICEncoder:
    name = "tfic"

    def __init__(self,
                 alpha: float = 1.0, beta: float = 1.0, eta: float = 1.0,
                 gamma_th: float = 0.5, kappa: float = 2.0,
                 gmax: int = 6, n_stages: int = 2, sweeps: int = 3,
                 c_cand: float = 8.0, top_m: int = 32,
                 max_tunnel_rows: int = 512, max_clusters_per_row: int = 50,
                 work_dtype=torch.float32):
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
        self.work_dtype = work_dtype

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def apply(self, state: IntegerQuantizedTensorState, stats: LayerStats):
        assert stats.Sigma is not None, (
            "TFIC needs a materialized G (use gram backend, keep_sigma=True; "
            "register 'tfic' in KEEP_SIGMA).")
        dev = state.scale.device
        wdt = self.work_dtype
        d = stats.d
        pin = state.padded_in_features

        # ---- build the exact reconstruction metric G = H = Sigma + mu mu^T
        mu = stats.mu_hat.to(device=dev, dtype=wdt)
        G = stats.Sigma.to(device=dev, dtype=wdt) + torch.outer(mu, mu)   # [d,d]
        if pin > d:
            Gp = torch.zeros(pin, pin, device=dev, dtype=wdt)
            Gp[:d, :d] = G
            idx = torch.arange(d, pin, device=dev)
            Gp[idx, idx] = torch.diagonal(G).mean()      # padded coords inert
            G = Gp
        diagG = torch.diagonal(G).contiguous()           # [pin]

        # ---- base state in spin space
        scale = state.scale.to(wdt)                      # [C,pin]; half-spacing H_ij = scale/2
        zp = state.zero_point.to(wdt)
        pre = state.pre_round.to(wdt)
        Wf = state.float_weights.to(wdt)                 # [C,pin] (W, padding-zeroed)
        Wint = state.integer_weights.to(wdt).clone()     # [C,pin] mutable codes
        C = Wint.shape[0]
        max_int = float(state.max_int)

        # residual R = Wq - W  (dequant minus float); RG = R @ G
        R = (Wint - zp) * scale - Wf                     # [C,pin]
        RG = R @ G                                        # [C,pin]

        e0 = (R * RG).sum().item()                        # exact starting energy Tr(R G R^T)

        # ---- top-m coupling neighbours of each column (by |G_jk|, off-diag)
        Gabs = G.abs().clone()
        Gabs.fill_diagonal_(0.0)
        m = min(self.top_m, pin - 1)
        nbr_val, nbr_idx = torch.topk(Gabs, m, dim=1)     # [pin,m] |G_jk|, indices
        del Gabs

        # boundary distance |D_ij|/H_ij in spin geometry: D = pre-round offset
        # |W/scale + zp - round(.)| in [0,0.5] -> 2*that in [0,1]; ambiguity = 1-2|frac|
        frac = (pre - Wint).abs().clamp(0, 0.5)           # distance to chosen level
        U_bnd = (1.0 - 2.0 * frac).clamp_min(0.0)         # [C,pin] in [0,1]

        total_flips = 0
        total_cluster_moves = 0
        cluster_energy = 0.0

        # ============================ annealing =========================== #
        # increasing pool thresholds, decreasing cluster caps across stages
        for a in range(self.n_stages):
            t = a / max(1, self.n_stages - 1)
            gamma_a = self.gamma_th * (0.6 + 0.4 * t)     # field threshold rises
            gmax_a = max(2, int(round(self.gmax - (self.gmax - 2) * t)))
            final_stage = (a == self.n_stages - 1)

            # ----- current single-flip geometry
            flip_dir = torch.sign(pre - Wint)
            flip_dir = torch.where(flip_dir == 0,
                                   torch.ones_like(flip_dir), flip_dir)
            proposed = Wint + flip_dir
            in_range = (proposed >= 0) & (proposed <= max_int)
            delta = flip_dir * scale                       # residual change on flip
            # single-flip gain dE_j (Eq. 15)
            dE = 2.0 * delta * RG + delta * delta * diagG.unsqueeze(0)
            dE = torch.where(in_range, dE, torch.full_like(dE, float("inf")))

            # ----- noise floor tau (Eq. 19): median |dE| over candidate spins
            finite = torch.isfinite(dE)
            if finite.any():
                tau = dE[finite].abs().median().item()
            else:
                tau = 0.0
            tau = max(tau, 1e-12)

            # ----- adaptive transverse field (Eq. 13)
            U_fld = torch.exp(-dE.clamp_min(0.0) / tau)    # energy mobility (inf->0)
            # frustration: sum_k (S_jk)+ via top-m neighbours, S_jk=-2 d_j d_k G_jk
            # column-level proxy on flip_dir signs of (c,j) vs (c,nbr)
            U_fru = torch.zeros_like(dE)
            G_nbr = G[torch.arange(pin, device=dev).unsqueeze(1), nbr_idx]  # [pin,m] G_jk
            for jrow in range(0, pin, 1024):               # chunk columns to bound mem
                je = min(jrow + 1024, pin)
                cols = slice(jrow, je)
                dj = delta[:, cols].unsqueeze(2)           # [C,nc,1]
                nidx = nbr_idx[cols]                        # [nc,m]
                dk = delta[:, nidx]                         # [C,nc,m]
                gjk = G_nbr[cols].unsqueeze(0)             # [1,nc,m]
                S = (-2.0 * dj * dk * gjk).clamp_min(0.0)  # [C,nc,m]
                U_fru[:, cols] = (S.sum(2) / tau).clamp(max=1.0)
                del dj, dk, gjk, S
            del G_nbr

            Gamma = self.alpha * U_bnd + self.beta * U_fld + self.eta * U_fru
            Gamma = torch.where(scale > 0, Gamma, torch.zeros_like(Gamma))
            pool = (Gamma > gamma_a) & in_range            # active spins U

            # ===================== Phase 1: descend ====================== #
            for _ in range(self.sweeps):
                flips_this_sweep = 0
                perm = torch.randperm(pin, device=dev)
                for j in perm.tolist():
                    if scale[0, j].item() == 0 and (scale[:, j] == 0).all():
                        continue
                    fdj = flip_dir[:, j]
                    dj = fdj * scale[:, j]                  # [C]
                    dEj = 2.0 * dj * RG[:, j] + dj * dj * diagG[j]
                    proposed_j = Wint[:, j] + fdj
                    okj = (proposed_j >= 0) & (proposed_j <= max_int) & pool[:, j]
                    # noise-floor acceptance (Eq. 20): dE < -kappa*tau
                    acc = okj & (dEj < -self.kappa * tau)
                    if not acc.any():
                        continue
                    nacc = int(acc.sum().item())
                    flips_this_sweep += nacc
                    total_flips += nacc
                    # apply: code += flip_dir on accepted rows of column j
                    step = torch.where(acc, fdj, torch.zeros_like(fdj))
                    Wint[:, j] = (Wint[:, j] + step).clamp(0, max_int)
                    dR = torch.where(acc, dj, torch.zeros_like(dj))    # [C]
                    R[:, j] = R[:, j] + dR
                    # rank-1 RG update (Eq. 18): RG += dR (outer) G[j,:]
                    RG += torch.outer(dR, G[j, :])
                    # refresh flip geometry for column j (sign may have changed)
                    fdj_new = torch.sign(pre[:, j] - Wint[:, j])
                    fdj_new = torch.where(fdj_new == 0,
                                          torch.ones_like(fdj_new), fdj_new)
                    flip_dir[:, j] = fdj_new
                if flips_this_sweep == 0:
                    break

            # ===================== Phase 2: tunnel ======================= #
            if final_stage or self.gmax < 2:
                continue
            # recompute single-flip gains at the local minimum
            flip_dir = torch.sign(pre - Wint)
            flip_dir = torch.where(flip_dir == 0,
                                   torch.ones_like(flip_dir), flip_dir)
            delta = flip_dir * scale
            proposed = Wint + flip_dir
            in_range = (proposed >= 0) & (proposed <= max_int)
            dE = 2.0 * delta * RG + delta * delta * diagG.unsqueeze(0)
            dE = torch.where(in_range, dE, torch.full_like(dE, float("inf")))

            # candidate spins pressed against a barrier: 0 <= dE <= c_cand*tau
            cand = in_range & (dE >= 0) & (dE <= self.c_cand * tau)
            cand_counts = cand.sum(1)
            rows = torch.nonzero(cand_counts >= 2, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            if rows.numel() > self.max_tunnel_rows:
                # prioritise rows with the most barrier-pressed candidates
                topr = torch.topk(cand_counts[rows], self.max_tunnel_rows).indices
                rows = rows[topr]

            for i in rows.tolist():
                cm, ce = self._tunnel_row(
                    i, Wint, R, RG, G, diagG, scale, flip_dir, pre,
                    cand[i], nbr_idx, dE[i], tau, gmax_a, max_int)
                total_cluster_moves += cm
                cluster_energy += ce

        # final exact energy
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
        del G, RG, R, scale, zp, pre, Wf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out.to(state.original_dtype), info

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _tunnel_row(self, i, Wint, R, RG, G, diagG, scale, flip_dir, pre,
                    cand_i, nbr_idx, dE_i, tau, gmax_a, max_int):
        """Synergy-clustered tunnelling on one output row i (paper section 3.7 Phase 2).

        Builds clusters by barrier synergy S_jk=-2 d_j d_k G_jk among candidate
        spins, enumerates all 2^|T| flip patterns, and accepts the minimizer by
        the EXACT group gain (Eq. 16). Updates Wint/R/RG in place. Returns
        (#accepted cluster moves, energy released)."""
        dev = Wint.device
        cand_cols = torch.nonzero(cand_i, as_tuple=False).flatten()
        if cand_cols.numel() < 2:
            return 0, 0.0
        cset = set(cand_cols.tolist())

        moves = 0
        released = 0.0
        clusters_built = 0

        # mutable per-row views
        # candidate ranking: lowest dE first = most "pressed" against barrier
        ranked = cand_cols[torch.argsort(dE_i[cand_cols])].tolist()

        used = set()
        for seed in ranked:
            if clusters_built >= self.max_clusters_per_row:
                break
            if seed in used:
                continue
            # grow cluster greedily by aggregate synergy with current members
            T = [seed]
            di_seed = (flip_dir[i, seed] * scale[i, seed]).item()
            while len(T) < gmax_a:
                best_k, best_syn = None, 0.0
                # neighbours of any current member that are candidates & unused
                for member in T:
                    nbrs = nbr_idx[member].tolist()
                    for k in nbrs:
                        if k in T or k in used or k not in cset:
                            continue
                        # synergy with the cluster: sum over members
                        dk = (flip_dir[i, k] * scale[i, k]).item()
                        syn = 0.0
                        for mm in T:
                            dm = (flip_dir[i, mm] * scale[i, mm]).item()
                            syn += -2.0 * dm * dk * G[mm, k].item()
                        if syn > best_syn:
                            best_syn, best_k = syn, k
                if best_k is None or best_syn <= 0.0:
                    break
                T.append(best_k)
            clusters_built += 1
            if len(T) < 2:
                continue

            # ---- exhaustive solve over 2^|T| flip patterns, exact group gain
            Tt = torch.tensor(T, device=dev)
            flip_T = flip_dir[i, Tt]                       # [|T|]
            scale_T = scale[i, Tt]
            delta_full = flip_T * scale_T                  # residual change if flipped
            G_TT = G[Tt][:, Tt]                            # [|T|,|T|]
            RG_T = RG[i, Tt]                               # [|T|]
            nT = len(T)
            # enumerate patterns f in {0,1}^nT
            best_gain, best_f = 0.0, None
            for mask in range(1, 1 << nT):
                f = torch.tensor([(mask >> b) & 1 for b in range(nT)],
                                 device=dev, dtype=scale.dtype)
                dT = f * delta_full                        # [|T|]
                # dE_T = 2<dT,RG_T> + dT^T G_TT dT  (Eq. 16)
                gain = (2.0 * (dT * RG_T).sum()
                        + dT @ (G_TT @ dT)).item()
                if gain < best_gain:
                    best_gain, best_f = gain, f
            if best_f is None or best_gain >= 0.0:
                continue

            # ---- accept: apply the cluster flip exactly
            dT = best_f * delta_full                        # [|T|]
            applied_dir = best_f * flip_T                    # integer code change
            Wint[i, Tt] = (Wint[i, Tt] + applied_dir).clamp(0, max_int)
            R[i, Tt] = R[i, Tt] + dT
            # RG update for row i only: RG[i,:] += dT @ G[Tt,:]
            RG[i, :] = RG[i, :] + dT @ G[Tt, :]
            moves += 1
            released += -best_gain
            used.update(T)

        return moves, released


def make_tfic(alpha=1.0, beta=1.0, eta=1.0, gamma_th=0.5, kappa=2.0,
              gmax=6, n_stages=2, sweeps=3, c_cand=8.0, top_m=32):
    return TFICEncoder(alpha=alpha, beta=beta, eta=eta, gamma_th=gamma_th,
                       kappa=kappa, gmax=gmax, n_stages=n_stages,
                       sweeps=sweeps, c_cand=c_cand, top_m=top_m)
