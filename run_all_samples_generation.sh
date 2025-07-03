#!/bin/bash

export SGLANG_API_KEY=lolgottem

MODEL_NAME="openrouter/deepseek/deepseek-r1-0528-qwen3-8b"

python3 scripts/generate_samples.py \
    run_name=test_hf_level_1 \
    dataset_src=huggingface \
    level=2 \
    num_workers=50 \
    server_type=litellm \
    model_name="$MODEL_NAME" \
    temperature=0.6 \
    runs_dir="runs/generation_logs_${MODEL_NAME}/" \
    log_prompt=True \
    verbose=True \
    "$@"