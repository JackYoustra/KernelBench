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
        ref_arch_src = state.input[0] if isinstance(state.input[0], str) else state.input[0].text

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

    problem_instruction = dedent(
        """
        Optimise the architecture named **Model** with custom CUDA operators! Name
        your optimised output architecture **ModelNew**. Write the code to a file
        called *model_new.py* in the sandbox so that the *bash* tool can see it.
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
# ReAct agent (tool loop) ------------------------------------------------------
# -----------------------------------------------------------------------------

react_agent = react(
    prompt=dedent(
        """
        You are a world‑class CUDA engineer working on an RTX‑5090 (sm_120)
        with CUDA 12.9. Use the available tools to craft an optimised *ModelNew*
        kernel. Remember to place the final code in *model_new.py*.
        """
    ),
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
    """Compose system guidance, prompt mutation, and the ReAct loop."""

    return chain(
        system_message(
            dedent(
                """
                You are a world‑class expert in CUDA programming.
                You have a 5090 manual mounted under */materials* and full
                internet access for reference.
                """
            )
        ),
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
