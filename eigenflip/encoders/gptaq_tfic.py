"""
GPTAQ + TFIC-A composite encoder.

The right division of labour (see notes in gptaq.py / tfica_fast.py):

  1. GPTAQ runs OBS sequential conditioning on the CONTINUOUS FP weight, with the
     asymmetric correction term passed through Hinv (twice). This absorbs the
     large-amplitude upstream-error correction into a shifted FP target `What`
     -- something a discrete flip pass cannot do, because flips are confined to
     +/-1 grid step and cannot reach a far target.

  2. TFIC-A then runs as a post-hoc rounding corrector, choosing floor/ceil
     AROUND What (not around the original W). Because GPTAQ already moved the FP
     target to the right place, the residual reconstruction field is small and
     within flip reach, so TFIC fixes the leftover rounding barriers without
     blowing up.

This is exactly the "GPTQ followed by TFIC" composition the TFIC paper proposes
in its Discussion, with GPTQ replaced by GPTAQ so the asymmetric (upstream-error)
correction is handled where it belongs -- in the continuous OBS update.

The TFIC stage here uses the SYMMETRIC objective Tr[R G Rᵀ] (field off): the
asymmetry has already been absorbed into What by GPTAQ, so re-injecting the K
field would double-count it. The encoder still measures R against What.
"""
from __future__ import annotations

import torch

from ..quantization.state import IntegerQuantizedTensorState
from ..statistics.trust_region import LayerStats
from .gptaq import GPTAQEncoder
from .tfic_fast import TFICEncoder as TFICEncoderFast


class GPTAQTFICEncoder:
    name = "gptaq_tfic"

    def __init__(self, damp=0.01, order="diag", alpha=0.25,
                 # TFIC hyper-params (symmetric flip pass)
                 tfic_alpha=1.0, tfic_beta=1.0, tfic_eta=1.0, gamma_th=0.5,
                 kappa=2.0, gmax=6, n_stages=2, sweeps=3, c_cand=8.0, top_m=32,
                 chunk_cols=256):
        self.gptaq = GPTAQEncoder(damp=damp, order=order, alpha=alpha)
        self.tfic = TFICEncoderFast(
            alpha=tfic_alpha, beta=tfic_beta, eta=tfic_eta, gamma_th=gamma_th,
            kappa=kappa, gmax=gmax, n_stages=n_stages, sweeps=sweeps,
            c_cand=c_cand, top_m=top_m, chunk_cols=chunk_cols)

    @torch.no_grad()
    def apply(self, state: IntegerQuantizedTensorState, stats: LayerStats):
        # --- stage 1: GPTAQ on the continuous FP weight -> What + final codes ---
        _out_gptaq, info_g = self.gptaq.apply(state, stats)
        What = info_g["What"].to(state.float_weights.device,
                                 state.float_weights.dtype)
        codes = info_g["codes"].to(state.float_weights.device)
        # pad What/codes back to padded width if needed (TFIC works in padded space)
        pin = state.padded_in_features
        if What.shape[1] < pin:
            Wp = torch.zeros(What.shape[0], pin, device=What.device,
                             dtype=What.dtype)
            Wp[:, :What.shape[1]] = What
            What = Wp
            Cp = torch.zeros(codes.shape[0], pin, device=codes.device,
                             dtype=codes.dtype)
            Cp[:, :codes.shape[1]] = codes
            codes = Cp

        # --- build a state centred on What, initialised from GPTAQ's FINAL
        # codebook (not a re-round of What) so TFIC starts from the committed
        # GPTAQ quantization and flips around the OBS target. ---
        gptq_state = IntegerQuantizedTensorState.from_gptq(
            What=What, scale=state.scale, zero_point=state.zero_point,
            max_int=state.max_int, group_size=state.group_size,
            in_features=state.in_features,
            padded_in_features=state.padded_in_features,
            original_dtype=state.original_dtype, codes=codes)

        # --- stage 2: TFIC symmetric flip around What with the ORIGINAL objective
        # Tr[R G Rᵀ] (field OFF: asymmetry already absorbed by GPTAQ into What/
        # codes; re-injecting K would double-count). ---
        out, info_t = self.tfic.apply(gptq_state, stats)
        info = {"encoder": self.name, "asymmetric": info_g.get("asymmetric"),
                "alpha": info_g.get("alpha"),
                "gptaq": {kk: vv for kk, vv in info_g.items()
                          if kk not in ("What", "codes")},
                "tfic": info_t}
        del What, codes, gptq_state
        return out, info


def make_gptaq_tfic(**kw):
    return GPTAQTFICEncoder(**kw)
