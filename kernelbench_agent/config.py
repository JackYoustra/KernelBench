# -*- coding: utf-8 -*-
"""
config.py - Centralized configuration for the KernelBench agent.
"""
from typing import Dict, Any

# --- Model & Project Configuration ---
MODEL_NAME: str = "KernelBench-Agent-v2"
BASE_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"  # Qwen models are excellent for tool use
PROJECT_NAME: str = "kernelbench-agent"

# --- LoRA & Unsloth Configuration ---
LORA_RANK: int = 32
MAX_SEQ_LENGTH: int = 8192 * 4 # 32768

# --- Environment & Execution Configuration ---
DOCKER_IMAGE: str = "kernelbench-sbx:latest"
HARDWARE: str = "5090"  # The hardware key for baseline lookup in results/
BASELINE_NAME: str = "baseline_time_torch" # The baseline file for speedup calculation

# --- ART Framework Internal Configuration ---
# This dictionary is passed to `art.TrainableModel` to idiomatically configure
# Unsloth's 4-bit loading, PEFT, and vLLM's memory-saving sleep mode.
INTERNAL_CONFIG: Dict[str, Any] = {
    "init_args": {
        "load_in_4bit": True,
        "max_seq_length": MAX_SEQ_LENGTH,
    },
    "peft_args": {
        "r": LORA_RANK,
        "lora_alpha": LORA_RANK * 2,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    },
    "engine_args": {
        "enable_sleep_mode": True
    },
}