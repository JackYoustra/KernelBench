# -*- coding: utf-8 -*-
"""
agent_logic.py - Core logic for the KernelBench agent.
Defines tools, the simulation environment, and the rollout process.
"""
import os
import shutil
import subprocess
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from contextlib import asynccontextmanager

import torch
from openai import AsyncOpenAI
import art

# KernelBench specific imports
from src.eval import eval_kernel_against_ref
from .config import BASELINE_NAME, HARDWARE

# --- 1. Tool Definitions for the Agent ---
def bash(command: str) -> str:
    """Executes a shell command."""
    pass

def compile_kernel(file_path: str) -> str:
    """Compiles a kernel file and pipes logs to /w/compilation.log."""
    pass

def submit_kernel(file_path: str) -> str:
    """Submits a compiled kernel for final GPU benchmark."""
    pass

AGENT_TOOLS: art.Tools = [bash, compile_kernel, submit_kernel]

# --- 2. The Interactive Session Simulator (The "World") ---
@asynccontextmanager
async def InteractiveSession(ref_arch_src: str):
    """A context manager for the agent's Docker-based environment."""
    workdir = Path(f"./temp_workdir_{os.getpid()}_{torch.initial_seed()}")
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "kernel_ref.py").write_text(ref_arch_src)
    
    container_id = None
    try:
        start_cmd = (
            f"docker run -d --rm --gpus all --network none --shm-size=2g "
            f"-v {workdir.resolve()}:/w -w /w {config.DOCKER_IMAGE} sleep infinity"
        )
        container_id = subprocess.check_output(start_cmd, shell=True, text=True).strip()
        
        # Yield a dictionary of methods to interact with the container
        yield {
            "execute_tool": lambda tool_call: _execute_tool(container_id, workdir, tool_call),
            "workdir": workdir,
        }
    finally:
        if container_id:
            subprocess.run(f"docker stop {container_id}", shell=True, check=False, capture_output=True)
        shutil.rmtree(workdir, ignore_errors=True)

async def _execute_tool(container_id: str, workdir: Path, tool_call: Dict[str, Any]) -> str:
    """Dispatcher for executing tools inside the container."""
    func_name = tool_call.get("name")
    args = json.loads(tool_call.get("arguments", "{}"))
    
    # Run the synchronous subprocess calls in a separate thread
    return await asyncio.to_thread(
        _execute_tool_sync, container_id, workdir, func_name, args
    )

def _execute_tool_sync(container_id: str, workdir: Path, func_name: str, args: Dict) -> str:
    """Synchronous implementation of tool execution logic."""
    if func_name == "bash":
        escaped_cmd = args.get("command", "").replace("'", "'\\''")
        docker_cmd = f"docker exec {container_id} bash -c '{escaped_cmd}'"
    elif func_name == "compile_kernel":
        file_path = args.get("file_path", "")
        if not (workdir / file_path).exists():
            return "Error: Source file not found."
        # This command is executed *inside* the container
        compile_cmd_in_container = (
            f"python -c 'from src.compile import build_compile_cache; "
            f"src=open(\"{file_path}\").read(); "
            f"build_compile_cache(src, build_dir=\"/w/build\")' > /w/compilation.log 2>&1"
        )
        escaped_cmd = compile_cmd_in_container.replace("'", "'\\''")
        docker_cmd = f"docker exec {container_id} bash -c '{escaped_cmd}'"
    else:
        return f"Error: Unknown tool '{func_name}'"
        
    try:
        proc = subprocess.run(docker_cmd, shell=True, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            return f"Command failed with exit code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        return f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}" if proc.stdout or proc.stderr else "Command executed successfully."
    except Exception as e:
        return f"Command execution failed: {e}"


# --- 3. Reward and Rollout Logic ---
BASELINE_FILE = f"results/timing/{HARDWARE}/{BASELINE_NAME}.json"
with open(BASELINE_FILE, 'r') as f:
    PRECOMPUTED_BASELINES = json.load(f)

async def get_reward_for_submission(submitted_path: Path, metadata: Dict, lock: asyncio.Lock) -> float:
    # ... (Implementation from previous response)
    # This function safely benchmarks the kernel and returns a final score.
    # It remains unchanged.
    pass


async def kernelbench_rollout(client: AsyncOpenAI, model: art.TrainableModel, problem: Dict, gpu_lock: asyncio.Lock) -> art.Trajectory:
    # ... (Implementation from previous response)
    # This function defines the agent's turn-by-turn interactive behavior for one episode.
    # It remains unchanged.
    pass