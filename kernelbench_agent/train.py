# -*- coding: utf-8 -*-
"""
train.py - Main script for training the KernelBench agent.
"""
import asyncio
import os
import subprocess

from datasets import load_dataset
import art
from art.local import LocalBackend

from . import config
from .agent_logic import kernelbench_rollout

async def main():
    print("Starting ART training process...")
    
    # Initialize the ART model using our centralized config
    model = art.TrainableModel(
        name=config.MODEL_NAME,
        project=config.PROJECT_NAME,
        base_model=config.BASE_MODEL,
        _internal_config=config.INTERNAL_CONFIG
    )
    
    backend = LocalBackend()
    await model.register(backend)
    
    client = model.openai_client()
    gpu_eval_lock = asyncio.Lock()
    
    dataset = load_dataset(config.DATASET_NAME, split="level_1", trust_remote_code=True).shuffle(seed=42)
    
    start_step = await model.get_step()
    print(f"Resuming training from step: {start_step}")
    
    for i in range(start_step, 1000):
        print(f"\n--- Starting training step {i+1} ---")
        
        # Use a slice of the dataset for each training step
        batch_size = 32
        problems_for_step = [p for p in dataset.select(range(i * batch_size, (i + 1) * batch_size))]
        
        if not problems_for_step:
            print("No more problems in the dataset. Ending training.")
            break
            
        rollout_tasks = [kernelbench_rollout(client, model, p, gpu_eval_lock) for p in problems_for_step]
        trajectories = await art.gather_trajectories(rollout_tasks, max_exceptions=batch_size)
        
        successful_trajectories = [t for t in trajectories if isinstance(t, art.Trajectory)]
        
        if not successful_trajectories:
            print("No successful trajectories in this batch. Skipping training step.")
            continue
            
        avg_reward = sum(t.reward for t in successful_trajectories) / len(successful_trajectories)
        print(f"Gathered {len(successful_trajectories)} trajectories. Average reward: {avg_reward:.2f}")
        
        await model.train([art.TrajectoryGroup(successful_trajectories)], config=art.TrainConfig(learning_rate=5e-5))
        print(f"--- Finished training step {i+1} ---")

if __name__ == "__main__":
    try:
        subprocess.run("docker info", shell=True, check=True, capture_output=True)
    except Exception:
        print("Docker is not running or not installed. Please start Docker for the agent simulator.")
        exit(1)
        
    asyncio.run(main())