# PYTHONPATH=. python eigenflip/run_gptaq.py \
#   --model-path /home/DATA/prometheus/anh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796  --output-dir ./quantized_models/gptaq_3bit \
#   --bits 3 --group-size 128 --asym-alpha 0.25 \
#   --calib-dataset c4 --n-calib 128 --seqlen 2048 \
#   --eval --eval-datasets wikitext2 c4 --eval-seqlen 2048 \
#   --delete-after-eval


# PYTHONPATH=. python eigenflip/run_gptaq.py \
#   --model-path /home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1 \
#   --output-dir ./quantized_models/gptaq_mistral_3bit \
#   --bits 3 --group-size 128 --asym-alpha 0.25 \
#   --calib-dataset c4 --n-calib 128 --seqlen 2048 \
#   --eval --eval-datasets wikitext2 c4 --eval-seqlen 2048 \
#   --delete-after-eval

# PYTHONPATH=. python eigenflip/run_gptaq.py \
#   --model-path /home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b \
#   --output-dir ./quantized_models/gptaq_llama_3bit \
#   --bits 3 --group-size 128 --asym-alpha 0.25 \
#   --calib-dataset c4 --n-calib 128 --seqlen 2048 \
#   --eval --eval-datasets wikitext2 c4 --eval-seqlen 2048 \
#   --delete-after-eval

PYTHONPATH=. python eigenflip/run_gptaq.py \
  --model-path /home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
  --output-dir ./quantized_models/gptaq_llama3_3bit \
  --bits 3 --group-size 128 --asym-alpha 0.25 \
  --calib-dataset c4 --n-calib 128 --seqlen 2048 \
  --eval --eval-datasets wikitext2 c4 --eval-seqlen 2048 \
  --delete-after-eval