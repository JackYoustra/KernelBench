"""KernelBench task definition that composes a custom CUDA‑prompting solver with
Inspect‑AI's built‑in ReAct agent. The agent runs with an *unlimited* cache so
all identical model calls are memoised across the evaluation.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from textwrap import dedent

from inspect_ai import Task, task
from inspect_ai.agent import react, agent
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.model import CachePolicy, get_model
from inspect_ai.solver import (
    chain,
    solver,
    system_message,
    Generate,
    TaskState,
    Solver,
)
from inspect_ai.scorer import Score, Target, Scorer
from inspect_ai.tool import bash, text_editor, think, web_search
from inspect_ai.util import sandbox

from src.eval import locked_eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from src.utils import extract_first_code, set_gpu_arch

# -----------------------------------------------------------------------------
# Scorer ----------------------------------------------------------------------
# -----------------------------------------------------------------------------


class KernelBenchScoreType(Enum):
    NOT_EXIST = "not_exist"
    NOT_EXTRACTED = "not_extracted"
    NOT_COMPILES = "not_compiles"
    NOT_CORRECT = "not_correct"


def kernelbench_score() -> Scorer:
    """Return a scorer that grades CUDA kernels for correctness and runtime."""

    gpu_semaphore = asyncio.Semaphore(1)  # only run one kernel at a time

    async def _score(state: TaskState, target: Target) -> Score:  # noqa: D401
        try:
            contents = await sandbox().read_file("model_new.py")
        except FileNotFoundError:
            return Score(value=KernelBenchScoreType.NOT_EXIST.value)

        custom_cuda = extract_first_code(contents, ["python", "cpp"])
        if custom_cuda is None:
            return Score(value=KernelBenchScoreType.NOT_EXTRACTED.value)

        # Reference kernel source is the task input (string or Blob)
        ref_arch_src = (
            state.input[0] if isinstance(state.input[0], str) else state.input[0].text
        )

        result = await locked_eval_kernel_against_ref(
            ref_arch_src,
            custom_cuda,
            verbose=True,
            measure_performance=True,
            num_correct_trials=5,
            num_perf_trials=100,
            gpu_semaphore=gpu_semaphore,
        )

        if not result.compiled:
            return Score(value=KernelBenchScoreType.NOT_COMPILES.value)
        if not result.correctness:
            return Score(value=KernelBenchScoreType.NOT_CORRECT.value)

        return Score(value=result.runtime)

    return _score


# -----------------------------------------------------------------------------
# Prompt‑mutation solver -------------------------------------------------------
# -----------------------------------------------------------------------------


@solver
def cudaify_prompt() -> Solver:
    """Inject CUDA‑specific optimisation instructions into the user prompt."""

    # ORIGINAL wording preserved verbatim
    problem_instruction = dedent(
        """
        Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in a file called model_new.py in the sandbox that your bash tool has access to.
        """
    )

    async def solve(state: TaskState, generate: Generate) -> TaskState:  # noqa: D401
        set_gpu_arch(["Blackwell"])
        prompt = state.user_prompt
        prompt.text = prompt_generate_custom_cuda_from_prompt_template(
            prompt.text, problem_instruction=problem_instruction
        )
        return state

    return solve


# -----------------------------------------------------------------------------
# Cached model wrapper ---------------------------------------------------------
# -----------------------------------------------------------------------------


@agent
async def cached_model(state, tools):  # type: ignore[override]
    """Thin agent that forwards to the evaluation model with an *infinite* cache."""

    state.output = await get_model().generate(
        state.messages,
        tools,
        cache=CachePolicy(expiry=None),  # never expire cache entries
    )
    state.messages.append(state.output.message)
    return state


# -----------------------------------------------------------------------------
# System messages (ORIGINAL wording) ------------------------------------------
# -----------------------------------------------------------------------------

NORMAL_SYSTEM_MESSAGE = system_message(
    dedent(
        """
        You are a world-class expert in CUDA programming.
        You are developing GPU kernels for a 5090 (sm_120) released in Feb 1, 2025.
        Your CUDA version is 12.9.
        You're developing on a container: nvcr.io/nvidia/pytorch:25.06-py3
        You're running on the same gpu you'll be tested on.
        You'll be graded on the correctness and performance of your final kernel.
        
        You have access to various tools to help you with your task.
        For the bash tool, there is a folder called materials that contains a manual for the 5090 and its requisite operations in markdown. Reference it to help you. You also have access to the internet if need be, and can use wget or curl in conjunction with your internet searches.
        You can perform any activities that you want in the contianer, including profiling, testing, reading, etc.
        """
    )
)

THINK_SYSTEM_MESSAGE = system_message(
    dedent(
        """
        Use the think tool to think about something. It will not obtain
        new information or make any changes to the repository, but just
        log the thought. Use it when complex reasoning or brainstorming
        is needed. For example, if you explore the repo and discover
        the source of a bug, call this tool to brainstorm several unique
        ways of fixing the bug, and assess which change(s) are likely to
        be simplest and most effective. Alternatively, if you receive
        some test results, call this tool to brainstorm ways to fix the
        failing tests.
        """
    )
)


# -----------------------------------------------------------------------------
# ReAct agent (tool loop) ------------------------------------------------------
# -----------------------------------------------------------------------------

# We rely on the system messages above; no additional prompt needed.
react_agent = react(
    tools=[
        bash(timeout=300),
        text_editor(timeout=300),
        think(),
        web_search(["openai", "anthropic", "gemini", "perplexity", "exa"]),
    ],
    attempts=5,
    model=cached_model,  # per‑call caching
)


# -----------------------------------------------------------------------------
# Top‑level solver chain -------------------------------------------------------
# -----------------------------------------------------------------------------


@solver
def kernelbench_solver() -> Solver:
    """Compose original system guidance, prompt mutation, and the ReAct loop."""

    return chain(
        NORMAL_SYSTEM_MESSAGE,
        THINK_SYSTEM_MESSAGE,
        cudaify_prompt(),
        react_agent,
    )


# -----------------------------------------------------------------------------
# Task definition --------------------------------------------------------------
# -----------------------------------------------------------------------------


@task
def kernelbench_task() -> Task:  # noqa: D401
    """Return the KernelBench Task object."""

    dataset = hf_dataset(
        "ScalingIntelligence/KernelBench",
        split="level_2",
        sample_fields=FieldSpec(
            input="code",
            id="problem_id",
            metadata=["name", "level"],
        ),
    )

    # Mount reference materials inside the sandbox
    for sample in dataset:
        sample.files = {"materials": "materials"}

    return Task(
        dataset=dataset,
        solver=kernelbench_solver(),
        sandbox="docker",
        scorer=kernelbench_score,
    )
