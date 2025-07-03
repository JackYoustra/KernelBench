#!/bin/bash
# Example configuration for generate_and_eval_single_sample.py

export SGLANG_API_KEY=lolgottem

MODEL_NAME="openrouter/google/gemini-2.5-pro"

uv run python scripts/generate_and_eval_single_sample.py \
    dataset_src="huggingface" \
    level=2 \
    problem_id=40 \
    gpu_arch="[Blackwell]" \
    server_type="litellm" \
    model_name="$MODEL_NAME" \
    temperature=0.6 \
    eval_mode="local" \
    logdir="results/eval_logs_${MODEL_NAME}" \
    log=True \
    log_prompt=True \
    log_generated_kernel=True \
    log_eval_result=True \
    verbose=True \
    "$@"  # Allow additional overrides from command line

    # top_p=0.95 \