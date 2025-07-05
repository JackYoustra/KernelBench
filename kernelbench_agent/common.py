# -*- coding: utf-8 -*-
"""
common.py - Shared logic for evaluation and reward calculation.

This module contains the core function for evaluating a submitted kernel against
the KernelBench benchmark. It is used by the training script to provide a reward
signal to the agent.
"""
import os
import json
import asyncio
from pathlib import Path
from typing import Dict

import torch

# KernelBench specific imports
from src.eval import eval_kernel_against_ref
from . import config

# --- Baseline Performance Data ---

# Load baselines once when the module is imported for efficiency.
# This avoids reading the file on every single reward calculation.
BASELINE_FILE = f"results/timing/{config.HARDWARE}/{config.BASELINE_NAME}.json"
try:
    with open(BASELINE_FILE, 'r') as f:
        PRECOMPUTED_BASELINES = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Baseline performance file not found at: {BASELINE_FILE}. "
        f"Please run `scripts/generate_baseline_time.py` for your hardware."
    )

# --- Reward Calculation Function ---

async def get_reward_for_submission(
    kernel_code: str,
    ref_arch_src: str,
    problem_metadata: Dict,
    build_dir: Path,
    gpu_lock: asyncio.Lock
) -> float:
    """
    Performs a locked, GPU-bound evaluation and returns a final reward score.
    This function implements a hierarchical reward system:
    1. A large penalty if the submitted code is empty or cannot be evaluated.
    2. A large penalty if the code fails to compile.
    3. A significant penalty if the code is incorrect.
    4. A positive base reward for correctness, plus an uncapped bonus
       proportional to the speedup achieved over the baseline.

    Args:
        kernel_code: The string content of the agent's submitted Python file.
        ref_arch_src: The string content of the original reference kernel.
        problem_metadata: A dictionary containing details like 'level' and 'problem_name'.
        build_dir: The path to the directory where compilation artifacts should be stored.
        gpu_lock: An asyncio.Lock to ensure serialized access to the GPU for benchmarking.

    Returns:
        A float representing the final reward for the trajectory.
    """
    if not kernel_code:
        return -15.0  # Penalize for submitting empty or unreadable code

    # Acquire the lock to ensure exclusive GPU access for benchmarking.
    async with gpu_lock:
        print(f"PID {os.getpid()}: Acquired GPU lock for benchmarking {problem_metadata['problem_name']}...")
        try:
            # Run the blocking, CPU/GPU-bound evaluation in a separate thread
            # to avoid blocking the main asyncio event loop.
            eval_result = await asyncio.to_thread(
                eval_kernel_against_ref,
                original_model_src=ref_arch_src,
                custom_model_src=kernel_code,
                measure_performance=True,
                num_correct_trials=3,
                num_perf_trials=20,
                device=torch.device("cuda:0"),
                build_dir=str(build_dir)
            )

            # --- Hierarchical Reward Logic ---
            if not eval_result.compiled:
                return -10.0
            if not eval_result.correctness:
                return -5.0

            # If correct, calculate performance reward
            level = problem_metadata["level"]
            problem_name = problem_metadata["problem_name"]
            baseline_stats = PRECOMPUTED_BASELINES[f"level{level}"].get(problem_name)

            if not baseline_stats or "mean" not in baseline_stats:
                # Correctness is met, but we can't score performance.
                # This is better than being incorrect, but not a full success.
                return 5.0

            baseline_time = baseline_stats["mean"]
            kernel_time = eval_result.runtime

            if baseline_time > 0 and kernel_time > 0:
                speedup = baseline_time / kernel_time
                # Base reward for correctness + uncapped speedup bonus
                return 10.0 + (speedup * 2.0)

            # Treat a timing failure (e.g., kernel_time <= 0) as a correctness failure
            return -5.0

        except Exception as e:
            print(f"Error during submission evaluation for {problem_metadata['problem_name']}: {e}")
            # Treat any other unexpected evaluation error as a compilation failure
            return -10.0
        finally:
            print(f"PID {os.getpid()}: Released GPU lock for {problem_metadata['problem_name']}.")