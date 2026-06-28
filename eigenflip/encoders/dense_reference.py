"""
Dense reference encoders (Section 6.5 validation harness).

These DO form a d x d matrix on purpose -- they are O(d^3) references, not the
deployed path. Two uses:

  DenseSurrogateGPTQ : run plain GPTQ/OBS sequential conditioning on the
                       MATERIALIZED H~_{k,eps} = D + V V^T. Algorithm 1 must
                       produce bitwise-identical codes to this. This is the
                       proof that EigenFlip Solve is an exact structured
                       implementation of the sequential rule, not an
                       approximation.

  DenseGPTQ          : the same sequential conditioning on the full empirical
                       second moment H (with diagonal damping) -- i.e. the
                       'gptq' ENCODER row of Table 2/6, runnable on any base.

Both consume the same IntegerQuantizedTensorState + LayerStats contract.
DenseGPTQ needs stats.H or stats.Sigma materialized (gram backend, keep_sigma).
"""

from __future__ import annotations

import torch

from ..quantization.state import IntegerQuantizedTensorState
from ..statistics.trust_region import LayerStats


@torch.no_grad()
def _robust_hinv(H, damp_percent=0.01, damp_auto_increment=0.0015):
    """Inverse-Hessian exactly as GPTQModel's hessian_inverse: scalar damping
    (damp_percent * mean(diag)), Cholesky-based inverse, auto-increment of damp
    on Cholesky failure, and a relative diagonal floor for singular blocks.
    Returns the FULL inverse H^{-1} (we run a dense reference, not the running
    Cholesky-factor downdate, so we want the explicit inverse).

    This replaces a naive torch.linalg.inv, which produces inf/NaN on the
    ill-conditioned deep-layer Hessians (observed: dirty stream -> NaN).
    """
    import math
    H = H.clone()
    diag_view = H.diagonal()
    orig_diag = diag_view.clone()
    base_abs_max = torch.max(orig_diag.abs()).item()
    if not math.isfinite(base_abs_max) or base_abs_max == 0.0:
        base_abs_max = 1.0
    floor_base = base_abs_max * 1e-6
    max_floor_attempts = 6

    attempt = 0
    while attempt <= max_floor_attempts:
        if attempt == 0:
            current_diag = orig_diag
        else:
            inc = floor_base * (10.0 ** (attempt - 1))
            current_diag = torch.clamp(orig_diag + inc, min=inc)
        diag_view.copy_(current_diag)
        mean = torch.mean(current_diag)
        damp = damp_percent
        while 0 < damp < 1:
            try:
                diag_view.add_(damp * mean)
                L = torch.linalg.cholesky(H)
                Hinv = torch.cholesky_inverse(L)          # full H^{-1}
                diag_view.copy_(current_diag)
                del L
                return Hinv
            except Exception:
                diag_view.copy_(current_diag)
                if damp_auto_increment != 0:
                    damp += damp_auto_increment
                else:
                    break
        attempt += 1
    # last resort: heavy diagonal load so inv is at least finite
    diag_view.copy_(orig_diag + (floor_base * 1e3))
    return torch.linalg.inv(H)


@torch.no_grad()
def _sequential_condition(Wf, scale, zp, lo, hi, Hmat, order, work_dtype,
                         dXXT=None, alpha=0.0, damp_percent=0.01):
    """
    GPTQ sequential conditioning under a dense quadratic Hmat, matching the
    coordinate order. Returns (codes [C,pin], What [C,pin]).

    `What` is the OBS-SHIFTED full-precision weight at the moment each column is
    quantized -- the continuous value the column is rounded FROM, AFTER all
    upstream-column compensations. This is the correct "pre-round" target for a
    post-hoc TFIC flip pass: TFIC chooses floor/ceil around What, NOT around the
    original W (rounding around W is a no-op once codes are committed).

    GPTAQ asymmetric calibration: pass `dXXT = ΔX·Xᵀ` ([pin,pin], NOT /n) and
    `alpha>0`. We build, in the permuted frame (where Hinv is already formed),
        P = alpha * (dXXT_perm @ Hinv^T).triu(1) @ Hinv
    and apply the extra term  + w_pre * P[i, i+1:]  to the remaining columns,
    reproducing GPTAQ's  W1[:, i:] -= err*Hinv[i,i:] - w*P[i,i:].
    With alpha=0 this is exactly plain GPTQ.
    """
    dev = Wf.device
    C, pin = Wf.shape
    Hmat = Hmat.to(work_dtype)
    order = list(order)

    p = torch.tensor(order, device=dev)
    Hp = Hmat.index_select(0, p).index_select(1, p)        # [pin, pin]
    Wp = Wf.index_select(1, p).clone()                     # [C, pin]
    sc_p = scale.index_select(1, p)                        # [C, pin]
    zp_p = zp.index_select(1, p)
    Hinv = _robust_hinv(Hp, damp_percent=damp_percent)     # GPTQModel-style
    codes_p = torch.empty(C, pin, device=dev, dtype=torch.long)
    What_p = torch.empty(C, pin, device=dev, dtype=work_dtype)

    Pp = None
    if dXXT is not None and alpha != 0.0:
        dX_p = (dXXT.to(work_dtype).index_select(0, p).index_select(1, p))
        # P = alpha * (dXXT @ Hinv^T).triu(1) @ Hinv   (GPTAQ Eq. for P)
        Pp = alpha * torch.triu(dX_p @ Hinv.t(), diagonal=1) @ Hinv
        del dX_p

    for i in range(pin):
        si = sc_p[:, i]; zpi = zp_p[:, i]
        w_pre = Wp[:, i].clone()             # OBS-shifted FP, BEFORE rounding
        What_p[:, i] = w_pre
        q = torch.clamp(torch.round(w_pre / si + zpi), lo, hi)
        w_dq = (q - zpi) * si
        e = w_pre - w_dq             # GPTQ sign: target - dequant
        codes_p[:, i] = q.long()
        if i + 1 < pin:
            denom = Hinv[i, i]
            factor = (Hinv[i, i+1:] / denom).to(work_dtype)         # [pin-i-1]
            Wp[:, i+1:] -= e.unsqueeze(1) * factor.unsqueeze(0)
            if Pp is not None:
                Wp[:, i+1:] += w_pre.unsqueeze(1) * Pp[i, i+1:].unsqueeze(0)
            col = Hinv[i+1:, i:i+1]
            row = Hinv[i:i+1, i+1:]
            Hinv[i+1:, i+1:] -= (col @ row) / denom

    codes = torch.empty(C, pin, device=dev, dtype=torch.long)
    What = torch.empty(C, pin, device=dev, dtype=work_dtype)
    codes.index_copy_(1, p, codes_p)
    What.index_copy_(1, p, What_p)
    del Hp, Wp, Hinv, codes_p, What_p, Pp
    return codes, What


class DenseSurrogateGPTQ:
    """GPTQ on the materialized H~_{k,eps} = D + V V^T. Reference for Algorithm 1."""
    name = "dense_surrogate_gptq"

    def __init__(self, order: str = "leverage", work_dtype=torch.float64):
        self.order = order
        self.work_dtype = work_dtype

    @torch.no_grad()
    def apply(self, state: IntegerQuantizedTensorState, stats: LayerStats):
        dev = state.scale.device
        wdt = self.work_dtype
        d = stats.d
        pin = state.padded_in_features

        D = stats.D.to(device=dev, dtype=wdt)
        V = stats.V.to(device=dev, dtype=wdt)
        if pin > d:
            Dp = torch.empty(pin, device=dev, dtype=wdt); Dp[:d] = D; Dp[d:] = D.mean()
            Vp = torch.zeros(pin, V.shape[1], device=dev, dtype=wdt); Vp[:d] = V
            D, V = Dp, Vp
        # MATERIALIZE the surrogate (reference only)
        Htilde = torch.diag(D) + V @ V.t()                 # [pin, pin]

        scale = state.scale.to(wdt); zp = state.zero_point.to(wdt)
        Wf = state.float_weights.to(wdt)
        lo, hi = float(state.min_int), float(state.max_int)

        order = self._order(D, V)
        codes, _What = _sequential_condition(Wf, scale, zp, lo, hi, Htilde, order, wdt)

        out = (codes.to(wdt) - zp) * scale
        if pin > d:
            out = out[:, :d]
        del Htilde, D, V
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out.to(state.original_dtype), {
            "encoder": self.name, "k": stats.k, "codes": codes}

    def _order(self, D, V):
        if self.order == "leverage":
            lev = (1.0 / D) * (V * V).sum(dim=1)
            return torch.argsort(lev, descending=True).tolist()
        if self.order == "diag":
            return torch.argsort(D, descending=True).tolist()
        return list(range(D.shape[0]))


class DenseGPTQ:
    """
    The 'gptq' ENCODER (Table 2 rung-4 row) on full H, diagonally damped.
    Runnable on any base. Requires stats.Sigma materialized (gram, keep_sigma);
    H = mu mu^T + Sigma. Damping: H + damp * diag(H).
    """
    name = "gptq"

    def __init__(self, damp: float = 0.01, order: str = "diag",
                 work_dtype=torch.float64):
        self.damp = damp
        self.order = order
        self.work_dtype = work_dtype

    @torch.no_grad()
    def apply(self, state: IntegerQuantizedTensorState, stats: LayerStats):
        assert stats.Sigma is not None, (
            "DenseGPTQ needs a materialized Sigma (use gram backend, "
            "keep_sigma=True).")
        dev = state.scale.device
        wdt = self.work_dtype
        d = stats.d
        pin = state.padded_in_features

        mu = stats.mu_hat.to(device=dev, dtype=wdt)
        H = stats.Sigma.to(device=dev, dtype=wdt) + torch.outer(mu, mu)      # [d, d]
        if pin > d:
            Hp = torch.zeros(pin, pin, device=dev, dtype=wdt)
            Hp[:d, :d] = H
            idx = torch.arange(d, pin, device=dev)
            Hp[idx, idx] = torch.diagonal(H).mean()
            H = Hp
        # diagonal damping (form ii)
        diagH = torch.diagonal(H).clone()
        H = H + self.damp * torch.diag(diagH)

        scale = state.scale.to(wdt); zp = state.zero_point.to(wdt)
        Wf = state.float_weights.to(wdt)
        lo, hi = float(state.min_int), float(state.max_int)

        if self.order == "diag":
            order = torch.argsort(diagH, descending=True).tolist()
        else:
            order = list(range(pin))
        codes, _What = _sequential_condition(Wf, scale, zp, lo, hi, H, order, wdt)

        out = (codes.to(wdt) - zp) * scale
        if pin > d:
            out = out[:, :d]
        del H, diagH
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out.to(state.original_dtype), {"encoder": self.name, "damp": self.damp}