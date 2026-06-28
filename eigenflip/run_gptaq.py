"""
run_gptaq.py -- standalone driver for the faithful GPTAQ reproduction
(eigenflip/gptaq_repro.py). Reuses ONLY calibration_utils. Does not touch the
state/stats/encoder machinery.

Example:
  PYTHONPATH=. python eigenflip/run_gptaq.py \
    --model-path $MODEL_PATH --output-dir ./quantized_models/gptaq_3bit \
    --bits 3 --group-size 128 --asym-alpha 0.25 \
    --calib-dataset c4 --n-calib 128 --seqlen 2048
"""
from __future__ import annotations
import argparse, os, json, shutil, subprocess, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eigenflip.gptaq_repro import gptaq_quantize_model

try:
    from calibration_utils import (get_c4_calibration_data,
                                   get_wikitext2_calibration_data)
except ImportError:
    get_c4_calibration_data = get_wikitext2_calibration_data = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", default="./quantized_models/gptaq")
    p.add_argument("--bits", type=int, default=3, choices=[2, 3, 4, 8])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--sym", action="store_true",
                   help="symmetric weight quant (default asymmetric, RTN-style)")
    p.add_argument("--w-clip", action="store_true",
                   help="MSE grid clip in find_params (reference args.w_clip)")
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--asym-alpha", type=float, default=0.25)
    p.add_argument("--no-act-order", action="store_true",
                   help="disable desc_act ordering (reference default is on)")
    p.add_argument("--calib-dataset", default="c4", choices=["c4", "wikitext2"])
    p.add_argument("--n-calib", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--cache-dir", default="./calibration_cache")
    p.add_argument("--seed", type=int, default=42)
    # --- eval / cleanup (mirrors run_all.sh [2/3], [2.5], [3/3]) ---
    p.add_argument("--eval", action="store_true",
                   help="run eval_ppl.py on the saved checkpoint after quantizing")
    p.add_argument("--eval-datasets", nargs="+", default=["wikitext2", "c4"],
                   choices=["wikitext2", "c4"])
    p.add_argument("--eval-seqlen", type=int, default=2048)
    p.add_argument("--delete-after-eval", action="store_true",
                   help="rm -rf the checkpoint dir after eval (ppl is preserved)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True).eval()

    if get_c4_calibration_data is None:
        raise RuntimeError("calibration_utils.py not importable")
    if args.calib_dataset == "c4":
        calib = get_c4_calibration_data(
            tok, n_samples=args.n_calib, seqlen=args.seqlen, seed=args.seed,
            return_tensors=True, cache_dir=args.cache_dir)
    else:
        calib = get_wikitext2_calibration_data(
            tok, n_samples=args.n_calib, seqlen=args.seqlen, seed=args.seed,
            cache_dir=args.cache_dir)

    print(f"GPTAQ repro: bits={args.bits} gs={args.group_size} "
          f"sym={args.sym} mse={args.w_clip} alpha={args.asym_alpha} "
          f"act_order={not args.no_act_order}")
    gptaq_quantize_model(
        model, calib, dev,
        w_bits=args.bits, groupsize=args.group_size, sym=args.sym,
        mse=args.w_clip, percdamp=args.percdamp, alpha=args.asym_alpha,
        act_order=not args.no_act_order, seqlen=args.seqlen, tokenizer=tok)

    out = os.path.join(args.output_dir, "rtn_gptaq")
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved -> {out}")

    if not args.eval:
        return

    # free the in-memory model before eval reloads the checkpoint
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- [2/3] eval_ppl on the checkpoint ----
    print(f">>> [2/3] eval_ppl on {out}")
    cmd = [sys.executable, "eval_ppl.py",
           "--model-path", out,
           "--datasets", *args.eval_datasets,
           "--seqlen", str(args.eval_seqlen)]
    subprocess.run(cmd, check=True)

    # ---- [2.5] preserve ppl.json next to the output dir ----
    src_ppl = os.path.join(out, "ppl.json")
    dst_ppl = os.path.join(args.output_dir, "rtn_gptaq_ppl.json")
    print(">>> [2.5] preserving ppl.json")
    try:
        shutil.copy(src_ppl, dst_ppl)
        print(f"    {dst_ppl}")
    except Exception as e:
        print(f"    WARN: could not preserve ppl.json: {e}")

    # ---- [3/3] delete the checkpoint dir ----
    if args.delete_after_eval:
        print(f">>> [3/3] deleting {out}")
        shutil.rmtree(out, ignore_errors=True)
    print("<<< done rtn+gptaq")


if __name__ == "__main__":
    main()
