#!/bin/bash

export SGLANG_API_KEY=lolgottem

MODEL_NAME="openrouter/deepseek/deepseek-r1-0528-qwen3-8b"

python3 scripts/eval_from_generations.py \
    gpu_arch="[Blackwell]" \
    run_name=test_hf_level_1 \
    dataset_src=local \
    level=2 \
    num_gpu_devices=1 \
    timeout=300 \
    build_cache=True \
    num_cpu_workers=7 \
    runs_dir="runs/generation_logs_${MODEL_NAME}/" \
    kernel_eval_build_dir="cache/kernel_eval_build_${MODEL_NAME}/" \
    "$@"
