PYTHONPATH=. python eigenflip/run_gptaq.py \
  --model-path /home/DATA/prometheus/anh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796  --output-dir ./quantized_models/gptaq_3bit \
  --bits 3 --group-size 128 --asym-alpha 0.25 \
  --calib-dataset c4 --n-calib 128 --seqlen 2048 \
  --eval --eval-datasets wikitext2 c4 --eval-seqlen 2048 \
  --delete-after-eval