"""
GPTAQ encoder -- a FAITHFUL port of GPTQModel's GPTAQ.quantize() + GPTQ.hessian_inverse().

This is a line-for-line port of the reference implementation
(Intelligent-Computing-Lab-Yale / GPTQModel ModelCloud), NOT a re-derivation:
  * hessian_inverse: scalar damping (damp_percent * mean(diag)), Cholesky-based
    inverse returned as an UPPER Cholesky FACTOR of H^{-1}, auto-increment damp
    recovery, and a relative diagonal floor for singular blocks.
  * quantize(): blocksize loop, dead-channel zeroing, desc_act ordering, and the
    GPTAQ asymmetric term  W1[:, i:] -= err1*Hinv1[i,i:] - w*P1[i,i:],
    P = alpha * (dXXT @ Hinv^T).triu(1) @ Hinv.

The ONLY additions over the reference:
  * we feed H and dXXT from collect_asym (cross-Gram) instead of GPTQModel hooks;
  * we use the state's pre-computed per-column scale/zero (RTN/AWQ grid) instead
    of an internal quantizer.find_params, since the eigenflip pipeline fixes the
    grid up front;
  * we record `What`, the OBS-shifted FP weight at the moment each column is
    rounded, so a downstream TFIC flip pass can round AROUND What.

H and dXXT MUST share the same normalisation (the reference uses
inp *= sqrt(2/nsamples) for BOTH), so that P = alpha*(dXXT Hinv^T).triu Hinv is
scale-correct. collect_asym provides H = Sigma + mu mu^T (= XXt/n) and the RAW
cross-Gram dXXT_raw = dX Xt = (X~-X) X^T (summed). We rescale dXXT_raw by 1/n to
match H's 1/n, which preserves the dXXT/H ratio P depends on.
"""
from __future__ import annotations

import math
import torch

from ..quantization.state import IntegerQuantizedTensorState
from ..statistics.trust_region import LayerStats


@torch.no_grad()
def _hessian_inverse(H, damp_percent=0.01, damp_auto_increment=0.0015):
    """Port of GPTQModel GPTQ.hessian_inverse. Returns (Hinv_factor, damp) where
    Hinv_factor is the UPPER Cholesky factor of H^{-1} (NOT the full inverse).
    The OBS loop uses this factor exactly as the reference does."""
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
                H2 = torch.linalg.cholesky(H)
                Hinv = torch.linalg.cholesky(torch.cholesky_inverse(H2), upper=True)
                diag_view.copy_(current_diag)
                del H2
                return Hinv, damp
            except Exception:
                diag_view.copy_(current_diag)
                if damp_auto_increment != 0:
                    damp += damp_auto_increment
                else:
                    break
        attempt += 1
    return None, 1.0


class GPTAQEncoder:
    name = "gptaq"

    def __init__(self, damp: float = 0.01, order: str = "diag",
                 alpha: float = 0.25, blocksize: int = 128,
                 damp_auto_increment: float = 0.0015, work_dtype=torch.float32):
        self.damp = damp
        self.order = order            # "diag" -> desc_act ordering, else natural
        self.alpha = float(alpha)
        self.blocksize = int(blocksize)
        self.damp_auto_increment = damp_auto_increment
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
        H = stats.Sigma.to(device=dev, dtype=wdt) + torch.outer(mu, mu)  # XXt/n
        if pin > d:
            Hp = torch.zeros(pin, pin, device=dev, dtype=wdt)
            Hp[:d, :d] = H
            idx = torch.arange(d, pin, device=dev)
            Hp[idx, idx] = torch.diagonal(H).mean()
            H = Hp

        n = max(1, int(getattr(stats, "n_samples", 0)) or 1)
        # cross-Gram dXXT, matched to H's 1/n normalisation (raw is a sum).
        have_asym = getattr(stats, "dXXT", None) is not None and self.alpha != 0.0
        if have_asym:
            dXXT = stats.dXXT.to(device=dev, dtype=wdt) / n
            if dXXT.shape[0] < pin:
                dP = torch.zeros(pin, pin, device=dev, dtype=wdt)
                dP[:dXXT.shape[0], :dXXT.shape[0]] = dXXT
                dXXT = dP
        else:
            dXXT = None

        scale = state.scale.to(wdt); zp = state.zero_point.to(wdt)
        Wf = state.float_weights.to(wdt)
        C = Wf.shape[0]
        if Wf.shape[1] < pin:
            Wpad = torch.zeros(C, pin, device=dev, dtype=wdt)
            Wpad[:, :Wf.shape[1]] = Wf
            Wf = Wpad
            sc = torch.ones(C, pin, device=dev, dtype=wdt); sc[:, :scale.shape[1]] = scale; scale = sc
            zpe = torch.zeros(C, pin, device=dev, dtype=wdt); zpe[:, :zp.shape[1]] = zp; zp = zpe
        lo, hi = 0.0, float(state.max_int)

        W = Wf.clone()

        # --- dead channels (reference: dead = diag(H)==0) ---
        dead = torch.diagonal(H) == 0
        if dead.any():
            H[dead, dead] = 1.0
            W[:, dead] = 0.0
            if dXXT is not None:
                dXXT[:, dead] = 0.0

        # --- desc_act ordering ---
        desc = (self.order == "diag")
        if desc:
            perm = torch.argsort(torch.diagonal(H), descending=True)
            W = W[:, perm]; H = H[perm][:, perm]
            scale = scale[:, perm]; zp = zp[:, perm]
            if dXXT is not None:
                dXXT = dXXT[perm][:, perm]
            invperm = torch.argsort(perm)

        Hinv, damp = _hessian_inverse(H, self.damp, self.damp_auto_increment)
        Qint = torch.zeros(C, pin, device=dev, dtype=torch.long)
        if Hinv is None:
            # reference raises; we RTN-fallback this layer to protect the stream.
            print(f"    [gptaq] WARN: Hessian not PD -> RTN fallback this layer")
            Qint = torch.clamp(torch.round(W / scale + zp), lo, hi).long()
            Q = (Qint.to(wdt) - zp) * scale          # dequantized
            What = W.clone()
        else:
            P = None
            if dXXT is not None:
                P = self.alpha * torch.triu(dXXT @ Hinv.t(), diagonal=1) @ Hinv
                del dXXT
            Q = torch.zeros_like(W)
            What = torch.zeros_like(W)
            cols = W.shape[1]
            bs = self.blocksize
            for i1 in range(0, cols, bs):
                i2 = min(i1 + bs, cols)
                count = i2 - i1
                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)             # DEQUANTIZED q (ref: quantizer.quantize)
                Qint1 = torch.zeros_like(W1, dtype=torch.long)  # integer codes (for TFIC)
                What1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]
                P1 = None if P is None else P[i1:i2, i1:i2]
                for i in range(count):
                    w = W1[:, i]
                    dcol = Hinv1[i, i]
                    si = scale[:, i1 + i]; zpi = zp[:, i1 + i]
                    What1[:, i] = w                         # OBS-shifted pre-round
                    # ref: q = self.quantizer.quantize(w)  -> DEQUANTIZED value
                    #      Q1[:, i] = q ;  err1 = (w - q) / d
                    q_int = torch.clamp(torch.round(w / si + zpi), lo, hi)
                    q = (q_int - zpi) * si                  # dequantized value (== ref q)
                    Q1[:, i] = q                            # store DEQUANTIZED (like ref)
                    Qint1[:, i] = q_int.long()
                    err1 = (w - q) / dcol
                    upd = err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    if P1 is not None:
                        upd = upd - w.unsqueeze(1).matmul(P1[i, i:].unsqueeze(0))
                    W1[:, i:] -= upd
                    Err1[:, i] = err1
                Q[:, i1:i2] = Q1
                Qint[:, i1:i2] = Qint1
                What[:, i1:i2] = What1
                upd2 = Err1.matmul(Hinv[i1:i2, i2:])
                if P is not None:
                    upd2 = upd2 - W1.matmul(P[i1:i2, i2:])
                W[:, i2:] -= upd2
            del Hinv
            if P is not None:
                del P

        # un-permute (ref: Q = Q[:, invperm])
        if desc:
            Q = Q[:, invperm]; Qint = Qint[:, invperm]
            What = What[:, invperm]
            scale = scale[:, invperm]; zp = zp[:, invperm]

        # ref returns Q as the DEQUANTIZED weight directly -- NO second dequant.
        out = Q
        if pin > d:
            out = out[:, :d]; What = What[:, :d]; Qint = Qint[:, :d]

        # final NaN guard: never let a bad layer poison the dirty stream.
        if not (torch.isfinite(out).all() and torch.isfinite(What).all()):
            print(f"    [gptaq] WARN: non-finite output -> RTN fallback")
            Wf0 = state.float_weights.to(wdt)
            sc0 = state.scale.to(wdt); zp0 = state.zero_point.to(wdt)
            q0 = torch.clamp(torch.round(Wf0 / sc0 + zp0), 0, state.max_int)
            Qint = q0.long()
            out = (q0 - zp0) * sc0
            What = Wf0

        info = {"encoder": self.name, "damp": damp, "alpha": self.alpha,
                "asymmetric": have_asym,
                "What": What.to(state.original_dtype),
                "codes": Qint.to(torch.long)}
        del H
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out.to(state.original_dtype), info


def make_gptaq(damp=0.01, order="diag", alpha=0.25):
    return GPTAQEncoder(damp=damp, order=order, alpha=alpha)
