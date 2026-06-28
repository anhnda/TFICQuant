"""
collect_asym.py -- BLOCK-CAUSAL collection for asymmetric calibration (TFIC-A).

Asymmetric counterpart of collect_fast.collect_and_encode_awq_style. The fast
AWQ path runs ONE full-precision forward and hooks every layer's input, so each
layer sees only the clean input X~ and never the quantized input X. Asymmetric
calibration (GPTAQ) needs the field shift

    F = (1/n) X~_sc^T dY^T,   dY = W_fp (X~ - X)     (per linear layer)

i.e. the layer-output deviation caused by the UPSTREAM accumulated input error
X~ - X (clean minus dirty), measured through the FP weight. This is the GPTAQ
asymmetric target: it corrects for errors injected by previously-quantized
layers, NOT for the current layer's own weight quantization (the TFIC quadratic
term already handles that). Using (W_fp - W_q) X~ instead is WRONG -- it double-
counts the current layer and degrades results.

GPTAQ's Algorithm 2 walks blocks in forward order keeping two activations:
  * X~ through the *unquantized* block (FP target),
  * X  through the *quantized* block   (deployed input),
quantizing each block before advancing. X~ is materialized only inside the
current block and dropped after F is folded -- never kept network-wide.

Per block, three cheap passes over the calib set, each touching ONE block:
  Pass A  clean stream through FP block       -> stream G on X~, save clean output
  Pass B  clean AND dirty streams through FP block -> fold
          F = (1/n) X~^T dY^T,  dY = W_fp (X~ - X)   (same FP weight!)
  final   encode WITH field F, write module.weight in place
  Pass C  dirty stream through quantized block -> advance X for next block

Memory (OOM): F is [d_in, C] (weight-sized), G is [d_in, d_in], both on
gram_device (CPU by default). Activations are transient per sample; clean/dirty
streams cached on CPU between blocks. No X~ kept across blocks.

Reset semantics: GPTAQ does NOT reset X globally (X accumulates quant error
through the network). What is recomputed/freed per block is X~. We replicate
that exactly.

Scope: decoder stacks with model.model.layers (LLaMA / Mistral / Qwen2).
"""

from __future__ import annotations

import gc
from typing import Callable

import torch
import torch.nn as nn
from tqdm import tqdm

from .trust_region import LayerStats, james_stein_mean


def is_lm_head(name: str) -> bool:
    return "lm_head" in name.lower()


def _get_decoder_layers(model):
    m = getattr(model, "model", model)
    if not hasattr(m, "layers"):
        raise RuntimeError(
            "collect_asym supports decoder stacks with model.model.layers "
            "(LLaMA/Mistral/Qwen). Use collect_fast for the symmetric path.")
    return m.layers


def _to_dev(kw, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kw.items()}


class _AsymAcc:
    """G on clean X~ (left factor), plus SF = sum_t x~ dy^T."""

    def __init__(self, d, C, device, gram_on_cpu=True):
        self.d = d
        self.C = C
        self.gram_device = torch.device("cpu") if gram_on_cpu else device
        self.s1 = torch.zeros(d, dtype=torch.float64, device=device)
        self.s2 = torch.zeros(d, dtype=torch.float64, device=device)
        self.n = 0
        self.G = torch.zeros(d, d, dtype=torch.float64, device=self.gram_device)
        # cross-Gram accumulator  Kacc = sum_t (x~_t - x_t) x_t^T  -> [d_in, d_in]
        # (= n * K^T with K = (1/n) X (X~ - X)^T). This is GPTAQ's dXXT, in pure
        # input space -- NO weight multiply. The encoder forms the field W K^T.
        self.Kacc = torch.zeros(d, d, dtype=torch.float64, device=self.gram_device)
        self.have_dx = False
        self._cached = None

    @torch.no_grad()
    def add_gram(self, x_tilde):
        xf = x_tilde.reshape(-1, x_tilde.shape[-1]).float()
        if xf.device != self.s1.device:
            xf = xf.to(self.s1.device, non_blocking=True)
        self.s1 += xf.sum(0).double()
        self.s2 += (xf * xf).sum(0).double()
        self.n += xf.shape[0]
        g = (xf.t() @ xf).double()
        self.G += g.to(self.gram_device)
        del g, xf

    @torch.no_grad()
    def add_cross(self, x_tilde, x_dirty):
        """Fold the input-space cross-Gram  Kacc += sum_t x_t (x~_t - x_t)^T
        into Kacc, i.e. A·ΔAᵀ (rows=tokens => xd.t() @ dx). Pure input space,
        NO weight. Verified: E_asym = Tr[R G Rᵀ] - 2 Tr[R K Wᵀ] with K=Kacc/n,
        K = (1/n) A ΔAᵀ, and the field shift is W Kᵀ."""
        xt = x_tilde.reshape(-1, x_tilde.shape[-1]).float()
        xd = x_dirty.reshape(-1, x_dirty.shape[-1]).float()
        if xd.device != xt.device:
            xd = xd.to(xt.device, non_blocking=True)
        dx = xt - xd                                  # ΔX = X~ - X   [n, d]
        kk = (xd.t() @ dx).double()                   # A·ΔAᵀ -> [d, d]
        self.Kacc += kk.to(self.gram_device)
        self.have_dx = True
        del kk, dx, xt, xd

    @torch.no_grad()
    def to_stats(self, k, eps, keep_sigma, eig_device, with_field):
        n = max(1, self.n)
        # cache the covariance: probe (no field) and final (with field) calls
        # share the SAME Sigma/eigh; building it twice was the hang.
        if getattr(self, "_cached", None) is None:
            mu = self.s1 / n
            diag_H = self.s2 / n
            mu_g = mu.to(self.G.device)
            Sigma = self.G / n - torch.outer(mu_g, mu_g)
            Sigma = 0.5 * (Sigma + Sigma.t())
            diag_Sigma = torch.diagonal(Sigma).clone()
            diag_H = diag_H.to(self.G.device)

            U_k = Lam_k = None
            # TFIC-A reads only Sigma (G = Sigma + mu mu^T). It never consumes
            # U_k/Lam_k, so for k<=0 we SKIP the eigh entirely -- a CPU fp64 eigh
            # on a d x d Gram (d~3.5k for Qwen2.5-7B) is minutes per call and was
            # the cause of the stall. Only compute it if a caller actually needs
            # the low-rank factors (k>0).
            if k > 0:
                S = Sigma if eig_device is None else Sigma.to(eig_device)
                evals, evecs = torch.linalg.eigh(S)
                topk = torch.argsort(evals, descending=True)[:k]
                Lam_k = evals[topk].clamp_min(0).to(Sigma.device)
                U_k = evecs[:, topk].to(Sigma.device)
                del evals, evecs
                if S is not Sigma:
                    del S
            self._cached = dict(mu_g=mu_g, diag_H=diag_H, diag_Sigma=diag_Sigma,
                                Sigma=Sigma, U_k=U_k, Lam_k=Lam_k)
        c = self._cached
        # field = W K^T, K = Kacc/n = (1/n) A ΔAᵀ  (input-space, [d,d]).
        # GPTAQ also needs the RAW cross-Gram dXXT = ΔX·Xᵀ = ΔA·Aᵀ = Kacc^T.
        K = (self.Kacc / n) if (with_field and self.have_dx) else None
        dXXT = (self.Kacc.t().contiguous()) if (with_field and self.have_dx) else None
        if K is not None:
            for nm, M in (("K", K), ("dXXT", dXXT)):
                bad = (~torch.isfinite(M)).sum().item()
                if bad:
                    print(f"    [asym] WARN: {bad} non-finite {nm} -> zeroed")
                    M.copy_(torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0))
            k_rms = K.pow(2).mean().sqrt().item()
            g_rms = c["Sigma"].pow(2).mean().sqrt().item() if c["Sigma"] is not None else float("nan")
            print(f"    [asym] |K|_rms={k_rms:.3e}  |Sigma|_rms={g_rms:.3e}")
        st = LayerStats(d=self.d, mu_hat=james_stein_mean(c["mu_g"]),
                        diag_H=c["diag_H"], diag_Sigma=c["diag_Sigma"],
                        U_k=c["U_k"], Lam_k=c["Lam_k"], eps=eps, F=K,
                        dXXT=dXXT, n_samples=n,
                        Sigma=c["Sigma"] if keep_sigma else None,
                        backend="gram_asym").build()
        return st

    def free(self):
        self.s1 = self.s2 = self.G = self.Kacc = self._cached = None


@torch.no_grad()
def collect_and_encode_asym(
    model, tokenizer, calib, device, *,
    k, eps, callback: Callable,
    keep_sigma=True,
    skip_lm_head=True,
    eig_on_cpu=False,
    gram_on_cpu=True,
    max_length=2048,
):
    """callback(name, module, LayerStats) must quantize + WRITE module.weight
    in place, exactly like run_fast.callback."""
    eig_device = torch.device("cpu") if eig_on_cpu else None
    layers = _get_decoder_layers(model)
    n_blocks = len(layers)
    print(f"  asymmetric block-causal: {n_blocks} decoder blocks")

    # ---- capture block-0 input (post-embedding) + block forward kwargs ----
    captured = []

    class _Catch(Exception):
        pass

    def catch_hook(_m, args, kwargs):
        hs = args[0] if args else kwargs.get("hidden_states")
        captured.append((hs.detach().to("cpu"),
                         {kk: (vv.detach().to("cpu") if torch.is_tensor(vv) else vv)
                          for kk, vv in kwargs.items() if kk != "hidden_states"}))
        raise _Catch

    h = layers[0].register_forward_pre_hook(catch_hook, with_kwargs=True)
    for sample in tqdm(calib, desc="  capture blk0", leave=False):
        try:
            if torch.is_tensor(sample):
                ids = sample.to(device)
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
            else:
                enc = tokenizer(sample, return_tensors="pt", truncation=True,
                                max_length=max_length)
                ids = enc["input_ids"].to(device)
            model(input_ids=ids, use_cache=False)
        except _Catch:
            pass
        except Exception:
            continue
    h.remove()

    kwargs_list = [kw for _, kw in captured]
    clean = [hs.clone() for hs, _ in captured]
    dirty = [hs.clone() for hs, _ in captured]
    del captured
    gc.collect()

    def _linears(block, prefix):
        return [(f"{prefix}.{n}", m) for n, m in block.named_modules()
                if isinstance(m, nn.Linear)
                and not (skip_lm_head and is_lm_head(n))]

    for bi in range(n_blocks):
        block = layers[bi]
        prefix = f"model.layers.{bi}"
        lin = _linears(block, prefix)
        print(f"\n[block {bi+1}/{n_blocks}] {len(lin)} linear layers")

        accs = {n: _AsymAcc(m.weight.shape[1], m.weight.shape[0], device,
                            gram_on_cpu=gram_on_cpu) for n, m in lin}
        fp_weight = {n: m.weight.data.clone() for n, m in lin}
        cap_in = {}

        def mk_pre(nm):
            def pre(_m, args, kwargs):
                x = args[0] if args else kwargs.get("input")
                cap_in[nm] = x.detach()
            return pre

        # ---- Pass A: clean stream through FP block; stream G; save outputs ----
        ha = [m.register_forward_pre_hook(mk_pre(n), with_kwargs=True)
              for n, m in lin]
        clean_out = []
        for si in tqdm(range(len(clean)), desc=f'  blk{bi} passA', leave=False):
            hs = clean[si].to(device)
            kw = _to_dev(kwargs_list[si], device)
            out = block(hs, **kw)
            out_hs = out[0] if isinstance(out, tuple) else out
            # NOTE: GPTAQ builds H from the DIRTY (deployed) input X, not the
            # clean X~. We therefore do NOT fold the Gram here; H is folded in
            # Pass B from the dirty stream (xd) so that Hinv matches the same
            # input distribution as dXXT = (X~-X) X^T. Folding H on X~ here made
            # Hinv mismatch the exploded dirty channels, blowing up
            # P = alpha (dXXT Hinv^T).triu Hinv -> non-finite -> RTN fallback.
            clean_out.append(out_hs.detach().to("cpu"))
            cap_in.clear()
            del hs, out, out_hs
        for hh in ha:
            hh.remove()

        # ---- Pass B: fold the input-space cross-Gram K = ΔX·Xᵀ (GPTAQ dXXT).
        # We need each layer's CLEAN input X~ and DIRTY input X. The asymmetric
        # term in E_asym = ||R X - W ΔX||^2 is -2 Tr[R K W^T] with K = (1/n) ΔA Aᵀ;
        # K is pure input-space (NO weight). The encoder forms the field W K^T.
        # This replaces the earlier (wrong) W(X~-X) "dY" formulation: GPTAQ keeps
        # the deviation in input space and lets the OBS update carry the weight.
        cap_clean, cap_dirty = {}, {}

        def mk_pre_into(store):
            def pre(_m, args, kwargs):
                x = args[0] if args else kwargs.get("input")
                store[id(_m)] = x.detach()
            return pre

        hb_c = [m.register_forward_pre_hook(mk_pre_into(cap_clean),
                                            with_kwargs=True) for _n, m in lin]
        for si in tqdm(range(len(clean)), desc=f'  blk{bi} passB', leave=False):
            kw = _to_dev(kwargs_list[si], device)
            hs_c = clean[si].to(device)
            block(hs_c, **kw)                        # FP block, clean input -> X~
            cap_dirty.clear()
            del hs_c
            hs_d = dirty[si].to(device)
            for hh in hb_c:
                hh.remove()
            hb_d = [m.register_forward_pre_hook(mk_pre_into(cap_dirty),
                                                with_kwargs=True) for _n, m in lin]
            block(hs_d, **kw)                        # FP block, dirty input -> X
            for hh in hb_d:
                hh.remove()
            hb_c = [m.register_forward_pre_hook(mk_pre_into(cap_clean),
                                                with_kwargs=True) for _n, m in lin]
            for n, m in lin:
                xt = cap_clean.get(id(m))            # X~ (clean)
                xd = cap_dirty.get(id(m))            # X  (dirty)
                if xt is None or xd is None:
                    continue
                accs[n].add_gram(xd)                 # H from DIRTY X (GPTAQ)
                accs[n].add_cross(xt, xd)            # fold ΔX·Xᵀ
            cap_clean.clear()
            cap_dirty.clear()
            del hs_d
        for hh in hb_c:
            hh.remove()

        # ---- final encode WITH field F: writes module.weight in place ----
        for n, m in tqdm(lin, desc=f'  blk{bi} final-q', leave=False):
            st = accs[n].to_stats(k, eps, keep_sigma, eig_device,
                                  with_field=True)
            callback(n, m, st)
            st.free_sigma()
            del st

        # ---- Pass C: advance dirty stream through quantized block ----
        for si in tqdm(range(len(dirty)), desc=f'  blk{bi} passC', leave=False):
            hs = dirty[si].to(device)
            kw = _to_dev(kwargs_list[si], device)
            out = block(hs, **kw)
            out_hs = out[0] if isinstance(out, tuple) else out
            dirty[si] = out_hs.detach().to("cpu")
            del hs, out, out_hs

        clean = clean_out                            # advance clean stream
        # ---- diagnostic: watch the two streams for blow-up (cascade) ----
        try:
            dmax = max(d.abs().max().item() for d in dirty)
            cmax = max(c.abs().max().item() for c in clean)
            ddev = max((dirty[i].to(torch.float32) - clean[i].to(torch.float32)
                        ).abs().max().item() for i in range(len(dirty)))
            nan_d = any((~torch.isfinite(d)).any().item() for d in dirty)
            print(f"  [diag blk{bi}] |dirty|max={dmax:.2f} |clean|max={cmax:.2f} "
                  f"|X~-X|max={ddev:.2f} dirty_nan={nan_d}")
        except Exception as _e:
            print(f"  [diag blk{bi}] failed: {_e}")
        for n in list(accs.keys()):
            accs[n].free()
        accs.clear()
        fp_weight.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  block {bi+1} done")