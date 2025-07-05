# -*- coding: utf-8 -*-
"""
train-kernelbench-agent-art.py

This script trains a tool-using agent for the KernelBench task using the `art`
framework and Unsloth. The agent is trained via reinforcement learning on
trajectories generated from an interactive, turn-by-turn process.

This final version reflects a sophisticated design:
- It uses the `art` framework for a robust, turn-by-turn RL training loop.
- The agent uses a lean, realistic toolset (`bash`, `compile_kernel`, `submit_kernel`).
- The `compile_kernel` tool pipes output to a log file, forcing the agent to learn a
  more realistic "run-then-check" workflow.
- A per-session compile cache is used to ensure absolute isolation and reproducibility.
- The GPU-bound benchmarking step is serialized with an `asyncio.Lock` to ensure
  reward signal integrity.
- The model is configured idiomatically via `art` to leverage Unsloth's 4-bit
  quantization and vLLM's memory-saving "sleep mode".

**SETUP BEFORE RUNNING:**

1.  **Project & ART Setup:**
    - This script must be in the root of the KernelBench project.
    - Create a directory `src/art/` and place all the provided `art`
      framework source files inside it.
    - Run `pip install -r requirements.txt` and `pip install -e .`.
    - Run `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`.
    - Run `pip install httpx openai "pydantic>=2.0" tqdm`.

2.  **Docker Setup:**
    - A Docker image named 'kernelbench-sbx:latest' must be built with all necessary
      dependencies (PyTorch, CUDA Toolkit, Ninja, etc.).

3.  **Baseline Timings:**
    - Ensure baseline timing files exist in `results/timing/` by running
      `scripts/generate_baseline_time.py`.
"""
import os
import shutil
import subprocess
import json
import asyncio
from pathlib import Path
from typing import Dict, Any

import torch
from datasets import load_dataset
from openai import AsyncOpenAI

# Import the ART framework and Unsloth
os.environ["IMPORT_PEFT"] = "1"
import art
from art.local import LocalBackend

# KernelBench specific imports
from src.eval import eval_kernel_against_ref

# --- Configuration Constants ---
DOCKER_IMAGE = "kernelbench-sbx:latest"
HARDWARE = "5090"
BASELINE_NAME = "baseline_time_torch_compile_cudagraphs"
DATASET_NAME = "ScalingIntelligence/KernelBench"
MODEL_NAME = "KernelBench-Agent-v2"
BASE_MODEL = "unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit"
PROJECT_NAME = "kernelbench-agent"
LORA_RANK = 32
MAX_SEQ_LENGTH = 8192 * 4 # 32768

# --- 1. Tool Definitions for the Agent ---
def bash(command: str) -> str:
    """
    Executes a shell command inside the /w directory of a sandboxed container.
    Use this to inspect files (ls, cat), and write or modify them (echo >).
    """
    pass

def compile_kernel(file_path: str) -> str:
    """
    Compiles the specified Python file containing a kernel. This is a CPU-bound
    operation. Logs are piped to /w/compilation.log.

    Args:
        file_path: The path to the Python file with the `ModelNew` class.

    Returns:
        A confirmation message. The agent must `cat` the log file to see results.
    """
    pass

def submit_kernel(file_path: str) -> str:
    """
    Submits the final, compiled kernel file for a GPU benchmark. This is a
    terminal action and concludes the optimization session.

    Args:
        file_path: The path to the Python file containing the optimized `ModelNew`.

    Returns:
        A JSON string with evaluation results (correctness, speedup).
    """
    pass

AGENT_TOOLS: art.Tools = [bash, compile_kernel, submit_kernel]

# --- 2. The Interactive Session Simulator (The "World") ---
class InteractiveSession:
    """Manages the Docker container and state for a single optimization attempt."""
    def __init__(self, ref_arch_src: str):
        self.workdir = Path(f"./temp_workdir_{os.getpid()}_{torch.initial_seed()}")
        self.workdir.mkdir(parents=True, exist_ok=True)
        (self.workdir / "kernel_ref.py").write_text(ref_arch_src)
        
        self.container_id = None
        start_cmd = (
            f"docker run -d --rm --gpus all --network none --shm-size=2g "
            f"-v {self.workdir.resolve()}:/w -w /w {DOCKER_IMAGE} sleep infinity"
        )
        self.container_id = subprocess.check_output(start_cmd, shell=True, text=True).strip()

    async def execute_tool(self, tool_call: Dict[str, Any]) -> str:
        """Asynchronously executes a tool call and returns the output."""
        func_name = tool_call.get("name")
        args = json.loads(tool_call.get("arguments", "{}"))
        
        if func_name == "bash":
            return await asyncio.to_thread(self._execute_bash, args.get("command", ""))
        elif func_name == "compile_kernel":
            return await asyncio.to_thread(self._compile_kernel, args.get("file_path", ""))
        else:
            return f"Error: Unknown tool '{func_name}' or malformed arguments."

    def _execute_bash(self, command: str) -> str:
        if not self.container_id: return "Error: Container not running."
        escaped_cmd = command.replace("'", "'\\''")
        docker_cmd = f"docker exec {self.container_id} bash -c '{escaped_cmd}'"
        proc = subprocess.run(docker_cmd, shell=True, capture_output=True, text=True, timeout=120)
        # check=True will raise an exception on non-zero exit codes
        if proc.returncode != 0:
            return f"Command failed with exit code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        return f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            
    def _compile_kernel(self, file_path: str) -> str:
        """Calls the CPU-bound compilation function, piping output to a log file."""
        kernel_src_path = self.workdir / file_path
        if not kernel_src_path.exists():
            return "Error: Source file not found for compilation."
        
        # This command runs inside the container via `docker exec`
        compile_cmd_in_container = (
            f"python -c 'from src.compile import build_compile_cache; "
            f"src=open(\"{file_path}\").read(); "
            f"build_compile_cache(src, build_dir=\"/w/build\")' > /w/compilation.log 2>&1"
        )
        self._execute_bash(compile_cmd_in_container)
        return "Compilation finished. Check `/w/compilation.log` for details."

    def cleanup(self):
        """Stops the container and removes the working directory."""
        if self.container_id:
            subprocess.run(f"docker stop {self.container_id}", shell=True, check=False, capture_output=True)
        shutil.rmtree(self.workdir, ignore_errors=True)

# Load baselines once
BASELINE_FILE = f"results/timing/{HARDWARE}/{BASELINE_NAME}.json"
with open(BASELINE_FILE, 'r') as f:
    PRECOMPUTED_BASELINES = json.load(f)

async def get_reward_for_submission(submitted_path: Path, metadata: Dict, lock: asyncio.Lock) -> float:
    """Performs the locked, GPU-bound evaluation and returns a final reward."""
    if not submitted_path.exists():
        return -15.0

    async with lock:
        print(f"PID {os.getpid()}: Acquired GPU lock for benchmarking {metadata['problem_name']}...")
        try:
            custom_kernel_src = await asyncio.to_thread(submitted_path.read_text)
            ref_arch_src, level, problem_name = metadata["ref_arch_src"], metadata["level"], metadata["problem_name"]
            
            # Run the blocking evaluation in a separate thread
            eval_result = await asyncio.to_thread(
                eval_kernel_against_ref,
                original_model_src=ref_arch_src, custom_model_src=custom_kernel_src,
                measure_performance=True, num_correct_trials=3, num_perf_trials=20,
                device=torch.device("cuda:0"), build_dir=str(submitted_path.parent / "build")
            )
            
            if not eval_result.compiled: return -10.0
            if not eval_result.correctness: return -5.0
            
            baseline_stats = PRECOMPUTED_BASELINES[f"level{level}"].get(problem_name)
            if not baseline_stats or "mean" not in baseline_stats: return -5.0
            
            baseline_time, kernel_time = baseline_stats["mean"], eval_result.runtime
            if baseline_time > 0 and kernel_time > 0:
                speedup = baseline_time / kernel_time
                return 10.0 + (speedup * 2.0)
            
            return -5.0
        except Exception as e:
            print(f"Error during submission evaluation: {e}")
            return -10.0
        finally:
            print(f"PID {os.getpid()}: Released GPU lock for {metadata['problem_name']}.")

# --- 3. The `rollout` Function: Generating Trajectories ---
async def kernelbench_rollout(client: AsyncOpenAI, model: art.TrainableModel, problem: Dict, gpu_lock: asyncio.Lock) -> art.Trajectory:
    """Simulates one full agent session for a given KernelBench problem."""
    
    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": "You are a world-class GPU kernel engineer..."},
            {"role": "user", "content": "Your task is to optimize the kernel in /w/kernel_ref.py..."}
        ],
        tools=AGENT_TOOLS,
        reward=0.0
    )
    
    session = InteractiveSession(ref_arch_src=problem["code"])
    
    try:
        for _ in range(15):  # Max 15 turns
            chat_completion = await client.chat.completions.create(
                messages=trajectory.messages(), model=model, tools=trajectory.tools, tool_choice="auto"
            )
            choice = chat_completion.choices[0]
            trajectory.messages_and_choices.append(choice)

            if not choice.message.tool_calls:
                trajectory.reward = -20.0
                break

            tool_call = choice.message.tool_calls[0]
            
            if tool_call.function.name == "submit_kernel":
                args = json.loads(tool_call.function.arguments)
                submitted_path = session.workdir.joinpath(args.get("file_path", ""))
                trajectory.reward = await get_reward_for_submission(submitted_path, problem, gpu_lock)
                break
            else:
                tool_output = await session.execute_tool(tool_call.function.to_dict())
                trajectory.messages_and_choices.append({
                    "role": "tool", "tool_call_id": tool_call.id, "content": tool_output
                })
        else:
            trajectory.reward = -25.0 # Failed to submit within max turns
    except Exception as e:
        print(f"Error during rollout: {e}")
        trajectory.reward = -30.0
    finally:
        session.cleanup()

    return trajectory.finish()

# --- 4. Main Training Loop ---
async def main():
    print("Starting ART training process...")

    # Define the internal config to enable 4-bit loading and sleep mode
    internal_config = {
        "init_args": {"load_in_4bit": True, "max_seq_length": MAX_SEQ_LENGTH},
        "peft_args": {"r": LORA_RANK, "lora_alpha": LORA_RANK * 2, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
        "engine_args": {"enable_sleep_mode": True},
    }
    
    model = art.TrainableModel(
        name=MODEL_NAME, project=PROJECT_NAME, base_model=BASE_MODEL, _internal_config=internal_config
    )
    backend = LocalBackend()
    await model.register(backend)
    
    client = model.openai_client()
    gpu_eval_lock = asyncio.Lock()
    
    dataset = load_dataset(DATASET_NAME, split="level_1", trust_remote_code=True).shuffle(seed=42)
    
    start_step = await model.get_step()
    print(f"Resuming training from step: {start_step}")
    
    for i in range(start_step, 1000):
        print(f"\n--- Starting training step {i+1} ---")
        batch_problems = [problem for problem in dataset.select(range(i*32, (i+1)*32))]
        
        rollout_tasks = [kernelbench_rollout(client, model, p, gpu_eval_lock) for p in batch_problems]
        trajectories = await art.gather_trajectories(rollout_tasks, max_exceptions=32)
        
        successful_trajectories = [t for t in trajectories if isinstance(t, art.Trajectory)]
        
        if not successful_trajectories:
            print("No successful trajectories in this batch. Skipping training step.")
            continue
            
        avg_reward = sum(t.reward for t in successful_trajectories) / len(successful_trajectories)
        print(f"Gathered {len(successful_trajectories)} trajectories for training. Average reward: {avg_reward:.2f}")
        
        await model.train([art.TrajectoryGroup(successful_trajectories)], config=art.TrainConfig(learning_rate=5e-5))
        print(f"--- Finished training step {i+1} ---")

if __name__ == "__main__":
    try:
        subprocess.run("docker info", shell=True, check=True, capture_output=True)
    except Exception:
        print("Docker is not running or not installed. Please start Docker to run the agent simulator.")
        exit(1)
        
    asyncio.run(main())