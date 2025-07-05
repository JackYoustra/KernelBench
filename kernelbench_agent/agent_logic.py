# -*- coding: utf-8 -*-
"""
agent_logic.py - Core logic for the KernelBench agent.
Defines tools, the simulation environment, and the rollout process.
"""
from contextlib import asynccontextmanager
import os
import shutil
import subprocess
import json
import asyncio
from pathlib import Path
from typing import Dict, Any

import torch
from openai import AsyncOpenAI
import art

# KernelBench specific imports
from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from . import config
from .common import get_reward_for_submission # Assuming common.py holds the reward function

# --- 1. Tool Definitions for the Agent ---
# The agent's toolset is simplified as requested.
def bash(command: str) -> str:
    """
    Executes a shell command inside the /w directory.
    """
    pass

def submit_kernel(file_path: str) -> str:
    """
    Submits the final kernel for a GPU benchmark for evaluation.
    """
    pass

AGENT_TOOLS: art.Tools = [bash, submit_kernel]

# --- 2. The Interactive Session Simulator (The "World") ---
@asynccontextmanager
async def InteractiveSession(ref_arch_src: str):
    """A context manager for the agent's Docker-based environment."""
    workdir = Path(f"./temp_workdir_{os.getpid()}_{torch.initial_seed()}")
    workdir.mkdir(parents=True, exist_ok=True)
    
    # Copy reference kernel and materials into the session directory
    (workdir / "kernel_ref.py").write_text(ref_arch_src)
    materials_path_src = Path("src/prompts/hardware")
    if materials_path_src.exists():
        shutil.copytree(materials_path_src, workdir / "materials")

    container_id = None
    try:
        start_cmd = (
            f"docker run -d --rm --gpus all --network none --shm-size=2g "
            f"-v {workdir.resolve()}:/w -w /w {config.DOCKER_IMAGE} sleep infinity"
        )
        container_id = subprocess.check_output(start_cmd, shell=True, text=True).strip()
        
        yield {
            "execute_tool": lambda tool_call: _execute_tool(container_id, tool_call),
            "workdir": workdir,
        }
    finally:
        if container_id:
            subprocess.run(f"docker stop {container_id}", shell=True, check=False, capture_output=True)
        shutil.rmtree(workdir, ignore_errors=True)

async def _execute_tool(container_id: str, tool_call: Dict[str, Any]) -> str:
    """Dispatcher for executing tools inside the container."""
    func_name = tool_call.get("name")
    args = json.loads(tool_call.get("arguments", "{}"))
    
    return await asyncio.to_thread(
        _execute_tool_sync, container_id, func_name, args
    )

def _execute_tool_sync(container_id: str, func_name: str, args: Dict) -> str:
    """Synchronous implementation of tool execution logic."""
    if func_name == "bash":
        command = args.get("command", "")
        escaped_cmd = command.replace("'", "'\\''")
        docker_cmd = f"docker exec {container_id} bash -c '{escaped_cmd}'"
    else:
        return f"Error: Unknown tool '{func_name}'"
        
    try:
        proc = subprocess.run(docker_cmd, shell=True, capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            return f"Command failed with exit code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        return f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}" if proc.stdout or proc.stderr else "Command executed successfully."
    except Exception as e:
        return f"Command execution failed: {e}"


# --- 3. Reward and Rollout Logic ---

async def kernelbench_rollout(client: AsyncOpenAI, model: art.TrainableModel, problem: Dict, gpu_lock: asyncio.Lock) -> art.Trajectory:
    """Simulates one full agent session for a given KernelBench problem."""

    # Use the standard KernelBench prompt constructor
    initial_user_prompt = prompt_generate_custom_cuda_from_prompt_template(problem["code"])

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": "You are a world-class GPU kernel engineer. Your goal is to optimize the provided kernel. Use the `bash` tool to write, compile, and test your code. When you are finished, use `submit_kernel` with the path to your final file.",
            },
            {"role": "user", "content": initial_user_prompt},
        ],
        tools=AGENT_TOOLS,
        reward=0.0
    )
    
    async with InteractiveSession(ref_arch_src=problem["code"]) as session:
        try:
            for _ in range(15):  # Max 15 turns per episode
                chat_completion = await client.chat.completions.create(
                    messages=trajectory.messages(), model=model, tools=trajectory.tools, tool_choice="auto"
                )
                choice = chat_completion.choices[0]
                trajectory.messages_and_choices.append(choice)

                if not choice.message.tool_calls:
                    trajectory.reward = -20.0  # Penalize for not calling a tool
                    break

                tool_call = choice.message.tool_calls[0]
                
                if tool_call.function.name == "submit_kernel":
                    args = json.loads(tool_call.function.arguments)
                    submitted_path = args.get("file_path", "")
                    
                    # Get the final code from the container before scoring
                    code_result_cmd = f"cat {submitted_path}"
                    code_result_str = _execute_tool_sync(session['container_id'], 'bash', {'command': code_result_cmd})
                    final_code = ""
                    if "STDOUT:" in code_result_str:
                        final_code = code_result_str.split("STDOUT:\n")[-1]

                    trajectory.reward = await get_reward_for_submission(
                        final_code, problem["code"], problem, session['workdir'] / "build", gpu_lock
                    )
                    break
                else:
                    tool_output = await session['execute_tool'](tool_call.function.to_dict())
                    trajectory.messages_and_choices.append({
                        "role": "tool", "tool_call_id": tool_call.id, "content": tool_output
                    })
            else:
                trajectory.reward = -25.0  # Failed to submit within max turns

        except Exception as e:
            print(f"Error during rollout: {e}")
            trajectory.reward = -30.0
    
    return trajectory.finish()