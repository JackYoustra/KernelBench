"""KernelBench task definition that composes a custom CUDA‑prompting solver with
Inspect‑AI's built‑in ReAct agent. The agent runs with an *unlimited* cache so
all identical model calls are memoised across the evaluation.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from textwrap import dedent
from typing import Any, List, Set
import hashlib

from inspect_ai import Task, task
from inspect_ai.agent import react, agent, Agent, AgentState, AgentAttempts, MessageFilter
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.model import trim_messages, ChatMessage
from inspect_ai.solver import (
    chain,
    solver,
    system_message,
    Generate,
    TaskState,
    Solver,
)
from inspect_ai.scorer import Score, Target, Scorer, scorer, accuracy, metric, Metric, SampleScore
from inspect_ai.tool import bash, text_editor, think, web_search, Tool
from inspect_ai.util import sandbox
import numpy as np

from src.eval import locked_eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from src.utils import extract_first_code, set_gpu_arch

# -----------------------------------------------------------------------------
# Scorer ----------------------------------------------------------------------
# -----------------------------------------------------------------------------


class KernelBenchScoreType(Enum):
    NOT_EXIST = "not_exist"
    NOT_COMPILES = "not_compiles"
    NOT_CORRECT = "not_correct"

# -- custom metric ------------------------------------------------------------

@metric
def mean_runtime_of_passes() -> Metric:
    """Average runtime across *successful* samples (ignores failures)."""

    def _metric(samples: list[SampleScore]) -> float:
        # When using metrics dict, the system extracts the specific key value
        # so s.score.value will be the runtime value (float or None), not the full dict
        runtimes = [
            s.score.value
            for s in samples
            if isinstance(s.score.value, (int, float)) and s.score.value is not None
        ]
        return float(np.mean(runtimes)) if runtimes else float("nan")

    return _metric


# -- scorer factory -----------------------------------------------------------

@scorer(
    metrics={
        "pass": [accuracy()],
        "runtime": [mean_runtime_of_passes()],
    }
)
def kernelbench_score() -> Scorer:
    """Scorer that returns either a *float* (runtime ⇒ success) or an enum.

    * Float → successful run; `pass=1`, `status="ok"`, `runtime=float`.
    * `KernelBenchScoreType` → failure code; `pass=0`, `runtime=None`.
    """

    gpu_semaphore = asyncio.Semaphore(1)

    async def _score(state: TaskState, target: Target) -> Score:  # noqa: D401
        # Helper that auto-derives status/pass/runtime from a single argument
        def _result(outcome: KernelBenchScoreType | float, metadata: dict[str, Any] | None = None):
            if isinstance(outcome, float):
                return Score(value={
                    "pass": 1,
                    "runtime": outcome,
                    "status": "ok",
                }, metadata=metadata)
            else:
                return Score(value={
                    "pass": 0,
                    "runtime": None,
                    "status": outcome.value,
                }, metadata=metadata)

        metadata = {}

        try:
            contents = await sandbox().read_file("model_new.py")
        except FileNotFoundError:
            # sometimes it also puts it in /model_new.py??? Maybe want to penalize it but we'll take it for now.
            try:
                contents = await sandbox().read_file("/model_new.py")
                metadata["contents_path"] = "/model_new.py"
            except FileNotFoundError:
                return _result(KernelBenchScoreType.NOT_EXIST)

        metadata["contents"] = contents

        custom_cuda = extract_first_code(contents, ["python", "cpp"])
        if custom_cuda is None:
            # just grab the contents of the file
            custom_cuda = contents

        
        ref_arch_src = state.metadata["code"]
        print("ref_arch_src")
        print(ref_arch_src)

        try:
            result = await locked_eval_kernel_against_ref(
                ref_arch_src,
                custom_cuda,
                verbose=True,
                measure_performance=True,
                num_correct_trials=5,
                num_perf_trials=100,
                gpu_semaphore=gpu_semaphore,
            )
        except Exception as e:
            metadata["result_compilation_error"] = str(e)
            return _result(KernelBenchScoreType.NOT_COMPILES, metadata=metadata)

        # Add result metadata and runtime stats with prefixes
        if hasattr(result, 'metadata') and result.metadata:
            for key, value in result.metadata.items():
                metadata[f"result_{key}"] = value
        
        if hasattr(result, 'runtime_stats') and result.runtime_stats:
            for key, value in result.runtime_stats.items():
                metadata[f"runtime_{key}"] = value

        if not result.compiled:
            return _result(KernelBenchScoreType.NOT_COMPILES, metadata=metadata)
        if not result.correctness:
            return _result(KernelBenchScoreType.NOT_CORRECT, metadata=metadata)

        return _result(result.runtime)

    return _score

async def incorrect_message(state: AgentState, scores: list[Score]) -> str:
    generic_error_message = "Your submission was incorrect. Please proceed and attempt to find the correct answer."
    last_value = scores[-1].value
    last_metadata = scores[-1].metadata
    if last_value["pass"] == 1:
        # ???
        return generic_error_message
    else:
        def errors_to_string() -> str:
            # grab compile error
            compile_log = last_metadata['result_compile_log'] if 'result_compile_log' in last_metadata else None
            compile_error = last_metadata['result_compilation_error'] if 'result_compilation_error' in last_metadata else None
            runtime_error = last_metadata['result_runtime_error'] if 'result_runtime_error' in last_metadata else None
            thing = ""
            if compile_log:
                thing += f"\nCompile Log: {compile_log}\n"
            if compile_error:
                thing += f"\nCompile Error: {compile_error}\n"
            if runtime_error:
                thing += f"\nRuntime Error: {runtime_error}\n" 
            return thing
        
        def mismatch_to_string() -> str:
            mismatch = last_metadata['correctness_issue']
            thing = ""
            if mismatch:
                thing += f"\nCorrectness Issue: {mismatch}\n"
            return thing
        
        # convert to the failure reason enum
        failure_reason = KernelBenchScoreType(last_value["status"])
        match failure_reason:
            case KernelBenchScoreType.NOT_EXIST:
                return "The file model_new.py (that would be revealed by ls model_new.py) does not exist. Please create it."
            case KernelBenchScoreType.NOT_COMPILES:                
                return f"The file model_new.py does not compile. Please fix the errors.{errors_to_string()}"
            case KernelBenchScoreType.NOT_CORRECT:
                return f"The file model_new.py is not correct. Please fix the errors.{errors_to_string()}{mismatch_to_string()}"
            case _:
                return generic_error_message

# -----------------------------------------------------------------------------
# Prompt‑mutation solver -------------------------------------------------------
# -----------------------------------------------------------------------------


@solver
def cudaify_prompt() -> Solver:
    """Inject CUDA‑specific optimisation instructions into the user prompt."""

    # ORIGINAL wording preserved verbatim
    problem_instruction = dedent(
        """
        Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in a file called model_new.py in the directory that your shell (and file IO tool) starts in - so writing to `model_new.py` will write to the correct file, while writing elsewhere (such as top level) will not.
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


# @agent
# def cached_model(state: AgentState, tools: list[Tool]) -> Agent:
#   async def _cached_model() -> AgentState:
#       """Thin agent that forwards to the evaluation model with an *infinite* cache."""

#       state.output = await get_model().generate(
#           state.messages,
#           tools,
#           cache=CachePolicy(expiry=None),  # never expire cache entries
#       )
#       state.messages.append(state.output.message)
#       return state

#   return _cached_model

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

        An example compile command used by pytorch in the inline compile process is:
        /usr/local/cuda/bin/nvcc --generate-dependencies-with-compile --dependency-output cuda.cuda.o.d -DTORCH_EXTENSION_NAME=model_kernels -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1016\" -isystem <redacted>/.venv/lib/python3.12/site-packages/torch/include -isystem <redacted>/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/local/cuda/include -isystem /usr/include/python3.12 -D_GLIBCXX_USE_CXX11_ABI=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_120,code=compute_120 -gencode=arch=compute_120,code=sm_120 --compiler-options '-fPIC' -std=c++17 -c <redacted>/.cache/torch_extensions/py312_cu128/model_kernels/cuda.cu -o cuda.cuda.o

        Note: DO NOT ADD CUSTOM GENCODE FLAGS TO THE INLINE PYTORCH COMMAND. ALL OF THE ABOVE FLAGS ARE AUTOMATICALLY DONE FOR YOU.
        
        You have access to various tools to help you with your task.
        For the bash tool, there is a read-only folder at /materials that contains a manual for the 5090 and its requisite operations in markdown.
        The contents of /materials/ is "ptx_isa_8.7.pdf" and it converted to a more easily readable format in /materials/conversion_results/ptx_isa_8.7.md.
        Associated images found in the md are also in the /materials/conversion_results/ directory.
        Reference it to help you. You also have access to the internet if need be, and can use wget or curl in conjunction with your internet searches.
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

# ---------------------------------------------------------------------------
# Context-window guard-rail: keep system + user, hash-compress old code dumps
# ---------------------------------------------------------------------------

# ----- helper ----------------------------------------------------------------
def _sha(text: str, n: int = 10) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:n]

def build_pruner(
    *,
    hash_dumps: bool = False,
    keep_all_summaries: bool = True,
    dump_tools: Set[str] | None = None,
    max_ctx_tokens: int = 130_000,
    preserve_ratio: float = 0.70,
):
    """
    hash_dumps          – hash-compress stale tool dumps
    keep_all_summaries  – True → keep every “[SUMMARY]: …” message
                          False (default) → keep only the newest one
    dump_tools          – names of tools whose outputs can be huge
    max_ctx_tokens      – per-request ceiling (< model ctx window)
    preserve_ratio      – 0.0-1.0 fraction to keep when over limit
    """

    if dump_tools is None:
        dump_tools = {
            "text_editor", "file_write",
            "text_editor.read", "text_editor.write",
            "file_read",
        }

    async def prune(messages: List["ChatMessage"]) -> List["ChatMessage"]:
        # indices we must preserve
        last_dump = max(
            (i for i, m in enumerate(messages)
             if m.tool_name in dump_tools and m.role == "assistant"),
            default=None,
        )
        summaries = [
            i for i, m in enumerate(messages)
            if m.role == "assistant"
            and isinstance(m.content_text, str)
            and m.content_text.lstrip().startswith("[SUMMARY]:")
        ]
        keep_summary_set = set(summaries if keep_all_summaries else summaries[-1:])

        kept = []
        for idx, msg in enumerate(messages):
            # always keep system, user, and *tool* role messages
            if msg.role in {"system", "user", "tool"}:
                kept.append(msg)
                continue

            # keep selected assistant messages
            if idx == last_dump or idx in keep_summary_set:
                kept.append(msg)
                continue

            # hash-compress stale dumps if enabled
            if (hash_dumps and msg.tool_name in dump_tools
                    and msg.role == "assistant"):
                placeholder = f"<SHA256:{_sha(msg.content_text)}:{len(msg.content_text)}B>"
                kept.append(msg._replace(content_text=placeholder))
                continue
            # else: drop the assistant message entirely

        # fail-safe trim if still too large
        from inspect_ai.model import count_tokens
        if count_tokens(kept) > max_ctx_tokens:
            kept = trim_messages(kept, preserve=preserve_ratio)

        return kept

    return prune

# -----------------------------------------------------------------------------
# ReAct agent (tool loop) ------------------------------------------------------
# -----------------------------------------------------------------------------

search_tool = web_search({
    "openai":       True,
    "anthropic":    True,
    "gemini":       True,
    "perplexity":   True,

    "exa": {"text": True}         # ← forces Exa to include the citation text field, will crash if not found
})

# We rely on the system messages above; no additional prompt needed.
react_agent = react(
    tools=[
        bash(timeout=300),
        text_editor(timeout=300),
        think(),
        search_tool,
    ],
    attempts=AgentAttempts(
        attempts=5,
        incorrect_message=incorrect_message,
    ),
    # model=cached_model,  # per‑call caching
    truncation=build_pruner(),
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
def kernelbench_task() -> Task:
    """Return the KernelBench Task object."""

    dataset = hf_dataset(
        "ScalingIntelligence/KernelBench",
        split="level_2",
        sample_fields=FieldSpec(
            input="code",
            id="problem_id",
            metadata=["name", "level", "code"],
        ),
    )

    return Task(
        dataset=dataset,
        token_limit=500_000,
        solver=kernelbench_solver(),
        sandbox="docker",
        scorer=kernelbench_score(),
    )
