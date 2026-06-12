# TFIC — Transverse-Field Ising Correction

Encoder cho post-training quantization (PTQ), chạy **trên base RTN** (3-bit,
group-size 128, Llama-3.1-8B). TFIC là một *encoder* ở tầng thứ 4 của pipeline
PTQ (transform → rate → codebook → **encoder**): nó **không** đổi scale/transform
của base, chỉ sửa các code nguyên (`Wint`) để hạ năng lượng tái dựng chính xác

```
E(s) = Tr(R G R^T),   R = (Wint - zp)·scale - Wf,   G = H = Σ + μμᵀ
```

bằng các bước lật spin (single-flip descent) + tunnelling theo cụm (group flip),
lấy cảm hứng từ mô hình Ising trường ngang.

> **Base = RTN, KHÔNG phải AWQ.** `run_all.sh` gọi `--base rtn --encoder
> tfic_fast`. AWQ chỉ là một base thay thế (`--base awq`, cần `--awq-scales-pt`)
> và **không** dùng ở đây. Lưu ý comment "AWQ-STYLE batched" trong `run_fast.py`
> chỉ nói về *lịch calibration* (gom N layer, calib 1 lần/batch cho nhanh —
> ~14 pass cho model 224-layer), **không** phải AWQ scaling. TFIC là correction
> độc lập, áp được trên bất kỳ base nào.

---

## Kết quả (Llama-3.1-8B, W3 / g128, calib c4 128×2048)

| Encoder (base RTN) | WikiText2 ↓ | C4 ↓ |
|---|---|---|
| `none` (RTN thuần)     | 11.0116 | 13.80   |
| `clc`                  | 9.4793  | 12.3363 |
| `eigenflip`            | 9.4395  | 12.2012 |
| **`tfic_fast`**        | **8.3946** | **11.0672** |

TFIC hạ WikiText2 từ 11.01 → **8.39** so với RTN base (−2.62), và vượt eigenflip
(−1.04). PPL đo bằng `eval_ppl.py` trên WikiText-2 và C4, seqlen 2048.

---

## Chạy

```bash
bash run_all.sh
```

`run_all.sh` lần lượt: (1) quantize RTN+encoder qua `run_fast.py`, (2) chấm PPL
qua `eval_ppl.py`, (3) lưu `rtn_<enc>_ppl.json` rồi xóa checkpoint. Sửa biến
`for ENC in ...` để chạy nhiều encoder. TFIC dùng `LBS=4` và `--eig-on-cpu`
(Gram-heavy).

Gọi trực tiếp một cell:

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

> Heavy GPU run + cần torch — chạy thủ công khi sẵn sàng, README không tự cài/chạy gì.

---

## Thuật toán (tóm tắt)

TFIC giữ nguyên nguyên tắc **"đề xuất xấp xỉ, chấp nhận chính xác"**: mọi nước đi
được verify lại bằng năng lượng tái dựng chính xác `E(s)=Tr(R G Rᵀ)`, đảm bảo
monotonic (E không bao giờ tăng).

**Phase 1 — descent (batched single-flip).** Với mỗi cột đề xuất một lần lật
spin tốt nhất (Eq. 15), nhận các cột vượt noise-floor theo chunk, cập nhật
`RG += dR_chunk @ G[chunk,:]` bằng **một** matmul (Eq. 18). Nếu một chunk làm
tăng năng lượng (do cross-term cùng hàng, Lemma 1) thì bisect — vẫn monotonic,
không sync per-column. Chunk size 1 ⇒ quy về sweep cột tham chiếu.

**Phase 2 — tunnelling (group flip).** Mọc cụm `T` theo synergy
`S_jk = -2 δ_j δ_k G_jk` quanh các cột candidate, enumerate `2^|T|` cấu hình lật,
nhận cụm có group-gain âm `dE_T = 2⟨δ_T, (RG)_{i,T}⟩ + δ_Tᵀ G_TT δ_T < 0`. Phần
scalar này chạy trên CPU/numpy để tránh `.item()` trong vòng lặp.

**Transverse field (Eq. 13).** `Γ = α·U_bnd + β·U_fld + η·U_fru` quyết định cột
nào vào pool; `U_fld` là field exp(−dE/τ), `U_fru` là frustration từ coupling
láng giềng top-m, `U_bnd` là biên rounding. `τ` = median |dE| (noise floor).

### Tham số chính

| Flag | Mặc định | Ý nghĩa |
|---|---|---|
| `--tfic-alpha/beta/eta` | 1.0 | trọng số 3 thành phần transverse field |
| `--tfic-gamma` | 0.5 | ngưỡng pool Γ |
| `--tfic-kappa` | 2.0 | ngưỡng chấp nhận `-κ·τ` |
| `--tfic-gmax` | 6 | kích thước cụm tối đa khi tunnel |
| `--tfic-stages` | 2 | số stage (anneal γ, gmax) |
| `--tfic-sweeps` | 3 | số sweep descent / stage |
| `--tfic-ccand` | 8.0 | cửa sổ candidate cho tunnel (`dE ≤ c·τ`) |
| `--tfic-topm` | 32 | số láng giềng coupling / cột |
| `--tfic-chunk` | 256 | số cột / chunk Phase 1 |

`tfic` (tham chiếu, per-column) và `tfic_fast` (batched) **cùng thuật toán, cùng
guarantee**; `tfic_fast` chỉ tái cấu trúc hot-loop để bỏ sync GPU↔CPU.

---

## Yêu cầu

- `keep_sigma=True` cho `tfic`/`tfic_fast` (cần G = Σ + μμᵀ vật chất hóa).
  Đã đăng ký trong `KEEP_SIGMA` của `run_fast.py`.
- torch + transformers + datasets; GPU cho quantize/eval (calib c4/wikitext2).
- `calibration_utils.py` import được (c4 / wikitext2 loader).

## File liên quan

```
eigenflip/encoders/tfic_fast.py   * encoder TFIC (batched) — file cốt lõi
eigenflip/encoders/tfic.py          encoder TFIC tham chiếu (per-column)
eigenflip/run_fast.py             * entry point: 1 base × 1 encoder / lần
eval_ppl.py                         PPL WikiText-2 / C4
run_all.sh                          driver: quantize → eval → lưu ppl.json
```