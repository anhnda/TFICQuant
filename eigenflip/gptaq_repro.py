"""
gptaq_repro.py -- a STANDALONE, line-for-line reproduction of the reference
GPTAQ implementation (GPTAQ-main/fake_quant/gptaq_utils.py), with NOTHING from
the eigenflip state/stats/encoder machinery. The only external dependency is the
calibration loader.

Reference semantics preserved exactly:
  * H and dXXT are accumulated in add_batch with the SAME running rescale
    inp *= sqrt(2/nsamples); H from the DIRTY input, dX = X~ - X from the FP
    cache, dXXT = dX @ X^T.
  * Per attention/MLP group the dirty stream `inps` is run through the (still-FP)
    block to collect H/dXXT; the clean stream `fp_inps` is run through the FP
    block with act-quant disabled to cache X~ per sub-layer.
  * fasterquant: scalar damp = percdamp*mean(diag(H)); Cholesky inverse ->
    upper Cholesky factor of H^{-1}; per-column OBS update with the GPTAQ
    asymmetric term  W1[:,i:] -= err1*Hinv1[i,i:] - w*P1[i,i:],
    P = alpha*(dXXT @ Hinv^T).triu(1) @ Hinv.
  * The per-column quantizer uses its OWN group-wise find_params on W (NOT a
    pre-baked RTN grid), exactly like the reference WeightQuantizer.

Weight is committed in place: layer.weight.data = Q (dequantized).
"""
from __future__ import annotations

import math
import logging

import torch
import torch.nn as nn
from tqdm import tqdm


# --------------------------------------------------------------------------- #
#  quant helpers (port of fake_quant/quant_utils.py asym/sym quant-dequant)
# --------------------------------------------------------------------------- #
def asym_quant(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(x, scale, maxq):
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


# --------------------------------------------------------------------------- #
#  WeightQuantizer (port of fake_quant/quant_utils.py WeightQuantizer)
# --------------------------------------------------------------------------- #
class WeightQuantizer(torch.nn.Module):
    def __init__(self, shape=1):
        super().__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(self, bits, perchannel=False, sym=True,
                  mse=False, norm=2.4, grid=100, maxshrink=.8):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym:
            self.maxq = torch.tensor(2 ** (bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2 ** bits - 1)

    def find_params(self, x):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1),
                                           zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)

    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


# --------------------------------------------------------------------------- #
#  GPTAQ (port of fake_quant/gptaq_utils.py GPTAQ)
# --------------------------------------------------------------------------- #
class GPTAQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.dXXT = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.fp_inp = []

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))

        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.dXXT *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        dX = self.fp_inp[0].float() * math.sqrt(2 / self.nsamples) - inp
        self.dXXT += dX.matmul(inp.t())

        del self.fp_inp[0]

    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1,
                    actorder=False, static_groups=False, alpha=0.25):
        W = self.layer.weight.data.clone()
        W = W.float()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        self.dXXT[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            self.dXXT = self.dXXT[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        Hinv = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)

        # scale it by alpha due to collection of dXXT and H
        P = alpha * ((self.dXXT @ Hinv.T).triu_(diagonal=1)) @ Hinv
        del self.dXXT

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            P1 = P[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(
                                W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)) \
                    - w.unsqueeze(1).matmul(P1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:]) - W1.matmul(P[i1:i2, i2:])

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning("NaN in weights")
            raise ValueError("NaN in weights")

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        self.dXXT = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
#  FP-input cache (port of fake_quant/model_utils.py FPInputsCache)
# --------------------------------------------------------------------------- #
class FPInputsCache:
    """Saves the clean (full-precision) per-sublayer input X~ via forward hooks."""

    def __init__(self, sequential):
        self.fp_cache = {}
        self.names = sum(sequential, [])
        for name in self.names:
            self.fp_cache[name] = []
        self.handles = []

    def cache_fp_input(self, m, inp, out, name):
        x = inp[0].detach()
        if len(x.shape) == 3:
            x = x.reshape((-1, x.shape[-1]))
        self.fp_cache[name] += [x.t()]

    def add_hook(self, full):
        import functools
        for name in self.names:
            self.handles.append(
                full[name].register_forward_hook(
                    functools.partial(self.cache_fp_input, name=name)))

    def clear_hook(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear_cache(self):
        for name in self.names:
            self.fp_cache[name] = []


# --------------------------------------------------------------------------- #
#  Block-causal driver (port of fake_quant/gptaq_utils.py gptaq_fwrd)
#  Generalized for LLaMA / Mistral / Qwen2 (model.model.layers), no
#  ActQuantWrapper needed: weight is committed by fasterquant in place, and the
#  clean cache is collected with the (still-FP) block before any commit.
# --------------------------------------------------------------------------- #
def _find_linears(module, prefix=""):
    res = {}
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear) and "lm_head" not in name:
            res[name] = child
    return res


@torch.no_grad()
def gptaq_quantize_model(model, calib, dev, *, w_bits=3, groupsize=128,
                         sym=False, mse=False, percdamp=0.01, alpha=0.25,
                         act_order=True, seqlen=2048, tokenizer=None):
    """
    calib: list of token-id tensors (or strings if tokenizer given).
    Mirrors gptaq_fwrd: capture block-0 input, then per block:
      A) FP forward of clean stream `fp_inps` (cache X~ per sublayer)
      B) per attn/MLP group: dirty forward of `inps` -> add_batch (H, dXXT)
         then fasterquant each sublayer
      C) advance dirty `inps` and clean `fp_inps` through the now-quantized /
         FP block respectively.
    """
    logging.info("-----GPTAQ Quantization-----")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    nsamples = len(calib)
    hidden = model.config.hidden_size

    inps = torch.zeros((nsamples, seqlen, hidden), dtype=dtype, device=dev)
    cache = {"i": 0, "kwargs": [None] * nsamples}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            # nn.Module intercepts attribute access; delegate anything we don't
            # define (e.g. Qwen2's decoder_layer.attention_type) to the wrapped
            # layer so the parent stack sees a normal decoder layer.
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.__dict__["_modules"]["module"], name)

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["kwargs"][cache["i"]] = {
                k: v for k, v in kwargs.items()}
            cache["i"] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    for b in calib:
        try:
            if torch.is_tensor(b):
                ids = b.to(dev)
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
            else:
                ids = tokenizer(b, return_tensors="pt").input_ids.to(dev)
            model(ids)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    kwargs_list = cache["kwargs"]

    # sequential groups (sublayers sharing an input), LLaMA/Mistral/Qwen layout
    sequential = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]

    fp_inps = inps.clone()

    def _kw(i):
        return {k: (v.to(dev) if torch.is_tensor(v) else v)
                for k, v in kwargs_list[i].items()}

    for i in range(len(layers)):
        print(f"\nLayer {i}:", flush=True, end=" ")
        layer = layers[i].to(dev)
        full = _find_linears(layer)
        # keep only names that actually exist in this layer
        seq = [[n for n in grp if n in full] for grp in sequential]
        seq = [grp for grp in seq if grp]

        fp_cache = FPInputsCache(seq)
        fp_cache.add_hook(full)
        for j in range(nsamples):
            fp_inps[j] = layer(fp_inps[j].unsqueeze(0), **_kw(j))[0]
        fp_cache.clear_hook()

        for names in seq:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                gptq[name] = GPTAQ(subset[name])
                gptq[name].quantizer = WeightQuantizer()
                gptq[name].quantizer.configure(
                    w_bits, perchannel=True, sym=sym, mse=mse)
                gptq[name].fp_inp = fp_cache.fp_cache[name]

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            first = list(subset.keys())[0]
            h = subset[first].register_forward_hook(add_batch(first))
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **_kw(j))[0]
            h.remove()

            for name in subset:
                if name != first:
                    gptq[name].H = gptq[first].H
                    gptq[name].dXXT = gptq[first].dXXT

            for name in subset:
                gptq[name].fasterquant(
                    percdamp=percdamp, groupsize=groupsize,
                    actorder=act_order, alpha=alpha)
                gptq[name].free()

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **_kw(j))[0]

        fp_cache.clear_cache()
        layers[i] = layer.cpu()
        del layer, gptq
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    logging.info("-----GPTAQ Quantization Done-----\n")
    return model