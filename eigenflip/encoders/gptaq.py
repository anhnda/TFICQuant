"""
GPTAQ encoder -- GPTQ with asymmetric calibration (Li et al., 2025).

Plain GPTQ conditions each column on the *quantized* upstream input; GPTAQ adds
a correction so the layer output matches the FULL-PRECISION upstream target. The
correction enters the OBS update as a second term  + w * P[i, i+1:]  with
    P = alpha * (dXXT @ Hinv^T).triu(1) @ Hinv,   dXXT = ΔX·Xᵀ, ΔX = X~ - X.
Crucially P passes through Hinv (twice): the asymmetric pull is normalised by the
Hessian, so it stays bounded even when ΔX is large at deep layers -- which is why
GPTAQ works on a continuous FP weight where a discrete-flip pass alone cannot.

This encoder also returns `What`, the OBS-shifted FP weight at the moment each
column is rounded. A downstream TFIC-A flip pass rounds around What (the GPTAQ
target), not around the original W, so the two stages compose:
    GPTAQ absorbs the large-amplitude asymmetric correction into the FP weight;
    TFIC then fixes the residual rounding barriers around that target, where the
    leftover field is small and within flip reach.

stats.F holds the input-space cross-Gram K = (1/n) A ΔAᵀ (from collect_asym), so
dXXT = ΔX·Xᵀ = n * K^T. We reconstruct dXXT = (stats.F * n).T.
"""
from __future__ import annotations

import torch

from ..quantization.state import IntegerQuantizedTensorState
from ..statistics.trust_region import LayerStats
from .dense_reference import _sequential_condition


class GPTAQEncoder:
    name = "gptaq"

    def __init__(self, damp: float = 0.01, order: str = "diag",
                 alpha: float = 0.25, work_dtype=torch.float64):
        self.damp = damp
        self.order = order
        self.alpha = float(alpha)
        self.work_dtype = work_dtype

    @torch.no_grad()
    def apply(self, state: IntegerQuantizedTensorState, stats: LayerStats):
        assert stats.Sigma is not None, (
            "GPTAQ needs a materialized Sigma (gram backend, keep_sigma=True).")
        dev = state.scale.device
        wdt = self.work_dtype
        d = stats.d
        pin = state.padded_in_features

        mu = stats.mu_hat.to(device=dev, dtype=wdt)
        H = stats.Sigma.to(device=dev, dtype=wdt) + torch.outer(mu, mu)
        if pin > d:
            Hp = torch.zeros(pin, pin, device=dev, dtype=wdt)
            Hp[:d, :d] = H
            idx = torch.arange(d, pin, device=dev)
            Hp[idx, idx] = torch.diagonal(H).mean()
            H = Hp
        diagH = torch.diagonal(H).clone()
        H = H + self.damp * torch.diag(diagH)

        scale = state.scale.to(wdt); zp = state.zero_point.to(wdt)
        Wf = state.float_weights.to(wdt)
        lo, hi = float(state.min_int), float(state.max_int)

        # asymmetric cross-Gram. collect_asym provides the RAW dXXT = ΔX·Xᵀ.
        dXXT = None
        alpha = 0.0
        if getattr(stats, "dXXT", None) is not None and self.alpha != 0.0:
            dM = stats.dXXT.to(device=dev, dtype=wdt)        # [d, d] raw sum
            if dM.shape[0] < pin:
                dP = torch.zeros(pin, pin, device=dev, dtype=wdt)
                dP[:dM.shape[0], :dM.shape[0]] = dM
                dM = dP
            dXXT = dM.contiguous()
            alpha = self.alpha
            del dM

        if self.order == "diag":
            order = torch.argsort(diagH, descending=True).tolist()
        else:
            order = list(range(pin))

        codes, What = _sequential_condition(Wf, scale, zp, lo, hi, H, order, wdt,
                                            dXXT=dXXT, alpha=alpha)

        out = (codes.to(wdt) - zp) * scale
        What_out = What
        if pin > d:
            out = out[:, :d]
            What_out = What[:, :d]
        info = {"encoder": self.name, "damp": self.damp, "alpha": alpha,
                "asymmetric": dXXT is not None,
                "What": What_out.to(state.original_dtype),  # OBS-shifted FP
                "codes": codes}
        del H, diagH, dXXT
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out.to(state.original_dtype), info


def make_gptaq(damp=0.01, order="diag", alpha=0.25):
    return GPTAQEncoder(damp=damp, order=order, alpha=alpha)
