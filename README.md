# TFIC — Transverse-Field Ising Correction

An encoder for post-training quantization (PTQ), running **on top of an RTN base**
(3-bit, group-size 128, Llama-3.1-8B). TFIC sits at the 4th stage of the PTQ
pipeline (transform → rate → codebook → **encoder**): it does **not** change the
base's scale/transform — it only edits the integer codes (`Wint`) to lower the
exact reconstruction energy

```
E(s) = Tr(R G R^T),   R = (Wint - zp)·scale - Wf,   G = H = Σ + μμᵀ
```

via spin-flip moves (single-flip descent) + cluster tunnelling (group flip),
inspired by the transverse-field Ising model.

> **Base = RTN, NOT AWQ.** `run_all.sh` invokes `--base rtn --encoder tfic_fast`.
> AWQ is only an alternative base (`--base awq`, requires `--awq-scales-pt`) and
> is **not** used here. Note the "AWQ-STYLE batched" comment in `run_fast.py`
> refers only to the *calibration schedule* (batch N layers, calibrate once per
> batch for speed — ~14 passes for a 224-layer model), **not** AWQ scaling. TFIC
> is an independent correction that can be applied on top of any base.

---

## Results (Llama-3.1-8B, W3 / g128, calib c4 128×2048)

| Encoder (base RTN) | WikiText2 ↓ | C4 ↓ |
|---|---|---|
| `none` (plain RTN)     | 11.0116 | 13.80   |
| `clc`                  | 9.4793  | 12.3363 |
| `eigenflip`            | 9.4395  | 12.2012 |
| **`tfic_fast`**        | **8.3946** | **11.0672** |

TFIC lowers WikiText2 from 11.01 → **8.39** vs. the RTN base (−2.62), and beats
eigenflip (−1.04). PPL measured by `eval_ppl.py` on WikiText-2 and C4, seqlen 2048.

---

## Running

```bash
bash run_all.sh
```

`run_all.sh` does, for each encoder: (1) quantize RTN+encoder via `run_fast.py`,
(2) score PPL via `eval_ppl.py`, (3) save `rtn_<enc>_ppl.json` then delete the
checkpoint. Edit the `for ENC in ...` line to run multiple encoders. TFIC uses
`LBS=4` and `--eig-on-cpu` (Gram-heavy).

Run a single cell directly:

```bash
PYTHONPATH=. python eigenflip/run_fast.py \
  --model-path <Llama-3.1-8B> \
  --output-dir ./quantized_models/eigenflip_3bit \
  --bits 3 --group-size 128 --k 16 \
  --base rtn --encoder tfic_fast \
  --calib-dataset c4 --n-calib 128 --seqlen 2048 \
  --layer-batch-size 4 --eig-on-cpu

PYTHONPATH=. python eval_ppl.py \
  --model-path ./quantized_models/eigenflip_3bit/rtn_tfic_fast \
  --datasets wikitext2 c4 --seqlen 2048
```

> Heavy GPU run + requires torch — run manually when ready; this README installs/runs nothing on its own.

---

## Algorithm (summary)

TFIC keeps the **"approximate proposals, exact acceptance"** principle: every move
is re-verified against the exact reconstruction energy `E(s)=Tr(R G Rᵀ)`,
guaranteeing monotonicity (E never increases).

**Phase 1 — descent (batched single-flip).** For each column, propose its single
best spin flip (Eq. 15), accept noise-floor-clearing columns per chunk, and update
`RG += dR_chunk @ G[chunk,:]` with **one** matmul (Eq. 18). If a chunk raises the
energy (same-row cross terms, Lemma 1), bisect it — still monotonic, no per-column
sync. Chunk size 1 reduces exactly to the reference column sweep.

**Phase 2 — tunnelling (group flip).** Grow a cluster `T` by synergy
`S_jk = -2 δ_j δ_k G_jk` around candidate columns, enumerate the `2^|T|` flip
configurations, and accept the cluster with negative group gain
`dE_T = 2⟨δ_T, (RG)_{i,T}⟩ + δ_Tᵀ G_TT δ_T < 0`. This scalar work runs on
CPU/numpy to avoid `.item()` inside the loop.

**Transverse field (Eq. 13).** `Γ = α·U_bnd + β·U_fld + η·U_fru` decides which
columns enter the pool; `U_fld` is the field exp(−dE/τ), `U_fru` is frustration
from top-m neighbour coupling, `U_bnd` is the rounding boundary. `τ` = median |dE|
(noise floor).

### Key parameters

| Flag | Default | Meaning |
|---|---|---|
| `--tfic-alpha/beta/eta` | 1.0 | weights of the 3 transverse-field terms |
| `--tfic-gamma` | 0.5 | pool threshold Γ |
| `--tfic-kappa` | 2.0 | acceptance threshold `-κ·τ` |
| `--tfic-gmax` | 6 | max cluster size when tunnelling |
| `--tfic-stages` | 2 | number of stages (anneal γ, gmax) |
| `--tfic-sweeps` | 3 | descent sweeps per stage |
| `--tfic-ccand` | 8.0 | candidate window for tunnelling (`dE ≤ c·τ`) |
| `--tfic-topm` | 32 | coupling neighbours per column |
| `--tfic-chunk` | 256 | columns per chunk in Phase 1 |

`tfic` (reference, per-column) and `tfic_fast` (batched) share the **same
algorithm and the same guarantees**; `tfic_fast` only restructures the hot loops
to remove GPU↔CPU syncs.

---

## Requirements

- `keep_sigma=True` for `tfic`/`tfic_fast` (needs a materialized G = Σ + μμᵀ).
  Already registered in `KEEP_SIGMA` in `run_fast.py`.
- torch + transformers + datasets; GPU for quantize/eval (calib c4/wikitext2).
- `calibration_utils.py` importable (c4 / wikitext2 loaders).

## Relevant files

```
eigenflip/encoders/tfic_fast.py   * TFIC encoder (batched) — core file
eigenflip/encoders/tfic.py          TFIC reference encoder (per-column)
eigenflip/run_fast.py             * entry point: 1 base × 1 encoder per run
eval_ppl.py                         PPL on WikiText-2 / C4
run_all.sh                          driver: quantize → eval → save ppl.json
```