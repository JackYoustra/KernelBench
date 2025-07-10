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
import os
import json
from functools import lru_cache

from inspect_ai import Task, task
from inspect_ai.agent import react, AgentState, AgentAttempts, MessageFilter, Agent
from inspect_ai.agent._types import ValueToFloat
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
from inspect_ai.scorer import Score, Target, Scorer, scorer, accuracy, metric, Metric, SampleScore, Value
from inspect_ai.tool import bash, text_editor, think, web_search
from inspect_ai.util import sandbox
import numpy as np
import polars as pl
from pathlib import Path

from src.eval import locked_eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from src.utils import extract_first_code, set_gpu_arch

# -----------------------------------------------------------------------------
# Load Baseline Data ----------------------------------------------------------
# -----------------------------------------------------------------------------

@lru_cache
def load_timing_data() -> pl.DataFrame:
    """Load all timing files into a single polars DataFrame"""
    path = Path(__file__).parent.parent / "results" / "timing" / "5090"
    BASELINE_FOLDER = os.environ.get("KERNELBENCH_BASELINE_FOLDER", path)
    if isinstance(BASELINE_FOLDER, str):
        BASELINE_FOLDER = Path(BASELINE_FOLDER)

    all_data = []
    
    # Walk through all JSON files in all subdirectories  
    for json_file in BASELINE_FOLDER.rglob("*.json"):
        # Extract config from parent directory name
        config = json_file.parent.name
        
        with open(json_file) as f:
            data = json.load(f)
        
        # Flatten each level into separate records
        for level_key, level_data in data.items():
            if level_key.startswith('level'):
                level_num = int(level_key[5:])  # Extract number from 'levelN'
                
                for name, timing_data in level_data.items():
                    # Each measurement becomes a row
                    if timing_data is None:
                        print(f"DEBUG: Timing data is None for {name}")
                        continue

                    all_data.append({
                        'level': level_num,
                        'name': name,
                        'config': config,
                        'file': json_file.name,
                        # type is name without the extension and without "baseline_time" at the start
                        'type': json_file.name.replace("baseline_time_", "").replace(".json", ""),
                        **timing_data  # mean, std, min, max, num_trials, hardware, device
                    })
    
    # Create DataFrame first to run checks on it
    df = pl.DataFrame(all_data)
    
    # checks!
    # assert that there are no duplicates with polars
    unique_combinations = df.select(pl.col("level", "name", "type")).unique()
    if unique_combinations.height != df.height:
        # Find duplicates for debugging
        duplicates = df.group_by(["level", "name", "type"]).len().filter(pl.col("len") > 1)
        print("Duplicate combinations found:")
        print(duplicates)
        
        # Show the actual duplicate rows
        for row in duplicates.iter_rows(named=True):
            level, name, type_val = row["level"], row["name"], row["type"]
            duplicate_rows = df.filter(
                (pl.col("level") == level) & 
                (pl.col("name") == name) & 
                (pl.col("type") == type_val)
            )
            print(f"\nDuplicate rows for level={level}, name={name}, type={type_val}:")
            print(duplicate_rows)
        
        assert False, f"Found {df.height - unique_combinations.height} duplicate rows"
    
    assert unique_combinations.height == df.height

    # assert that each sample that exists in one exists in all configs
    # Get all unique (level, name) pairs and all unique configs
    level_name_pairs = df.select(pl.col("level", "name")).unique()
    all_configs = df.select("config").unique().get_column("config").to_list()
    
    # For each (level, name) pair, ensure it exists in all configs
    for row in level_name_pairs.iter_rows(named=True):
        level, name = row["level"], row["name"]
        existing_configs = df.filter(
            (pl.col("level") == level) & (pl.col("name") == name)
        ).select("config").unique().get_column("config").to_list()
        
        missing_configs = set(all_configs) - set(existing_configs)
        assert not missing_configs, f"Sample (level={level}, name={name}) missing configs: {missing_configs}"
    
    return df

baselines = load_timing_data()

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
        # Extract runtime values, handling both dict and scalar cases
        runtimes = []
        for s in samples:
            value = s.score.value

            # Handle dictionary case - extract the 'runtime' key
            if isinstance(value, dict):
                runtime_val = value.get("runtime")
                if isinstance(runtime_val, (int, float)) and runtime_val is not None:
                    runtimes.append(runtime_val)
            # Handle scalar case (when value is already extracted)
            elif isinstance(value, (int, float)) and value is not None:
                runtimes.append(value)

        return float(np.mean(runtimes)) if runtimes else float("nan")

    return _metric


# -- scorer factory -----------------------------------------------------------

def _pass_metric(x: Value) -> float:
    if isinstance(x, dict):
        return x.get("pass", 0)
    elif isinstance(x, str):
        return 1 if x == "ok" else 0
    elif isinstance(x, int):
        return x
    elif isinstance(x, float):
        return x
    else:
        print(f"Unknown value type: {type(x)} for pass metric {x}")
        return 0

def _always_fail_metric(x: Value) -> float:
    return 0

pass_metric: ValueToFloat = _pass_metric
always_fail_metric: ValueToFloat = _always_fail_metric

@scorer(
    metrics={
        "pass": [accuracy(pass_metric)],
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

        # log the submission in the metadata
        state.metadata["submission_count"] = state.metadata.get("submission_count", 0) + 1
        metadata[f"submission_{state.metadata['submission_count']}"] = custom_cuda

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
        
        metadata["absolute_runtime"] = result.runtime

        # we have the absolute runtime
        # we want to adjust it to be relative to the reference
        # Calculate speedup using precomputed baseline
        level = state.metadata["level"]
        name = state.metadata["name"]

        assert level is not None and name is not None

        # log every baseline type to metadata
        baseline_rows = baselines.filter(pl.col("level") == level, pl.col("name") == name)
        
        # Log each baseline mean
        baseline_means = []
        for row in baseline_rows.iter_rows(named=True):
            baseline_type = row["type"]
            baseline_mean = row["mean"]
            
            # Log just the mean for each baseline type
            metadata[f"baseline_{baseline_type}_mean"] = baseline_mean
            baseline_means.append(baseline_mean)

        # Use the best (lowest) baseline time for speedup calculation
        best_baseline_time = min(baseline_means)
        kernel_time = result.runtime
        assert best_baseline_time > 0 and kernel_time > 0
        
        speedup = best_baseline_time / kernel_time
        metadata["speedup_best_torch"] = speedup
        metadata["best_baseline_time"] = best_baseline_time
        return _result(speedup, metadata=metadata)

    return _score

async def incorrect_message(state: AgentState, scores: list[Score]) -> str:
    generic_error_message = "Your submission was incorrect. Please proceed and attempt to find the correct answer."
    last_value = scores[-1].value
    last_metadata = scores[-1].metadata
    if last_value["pass"] == 1:
        # Agent success message with detailed runtime feedback
        runtime = last_metadata["absolute_runtime"]
        speedup = last_metadata["speedup_best_torch"]
        best_baseline = last_metadata["best_baseline_time"]
        
        # Collect all baseline runtimes
        baseline_runtimes = []
        for key, value in last_metadata.items():
            if key.startswith("baseline_") and key.endswith("_mean"):
                baseline_type = key.replace("baseline_", "").replace("_mean", "")
                baseline_runtimes.append(f"{baseline_type}: {value:.6f}s")
        
        baseline_summary = ", ".join(baseline_runtimes) if baseline_runtimes else "no baselines found"
        
        return (f"Your submission is correct! You achieved {runtime:.6f}s runtime. "
                f"The baseline runtimes are [{baseline_summary}] for a total score of "
                f"{speedup:.2f}x improvement when compared against the best baseline "
                f"runtime of {best_baseline:.6f}s. Attempt to improve your runtime further."
                "Available at your disposal are making your own quick tests by running the code in python, using ncu, using the torch profiler, using clang tidy (and other profiling tools, of course) and, of course, using your own thinking."
        )
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
        Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in a file called /workspace/model_new.py in the directory that your shell (and file IO tool) starts in. Note that your file IO tool can't take relative paths, you must write to the absolute path.  - so writing to `/workspace/model_new.py` will write to the correct file. You start in the `/workspace` directory.
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
# I hacked the dep file for now :p
# probably shoudln't

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
        Note that your bash tool has a limited context window. It may make sense to pipe the output of potentially large commands (such as, but not including, profilers) to a file and then read, grep, or perform other operations on the file.
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
) -> MessageFilter:
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

    async def prune(messages: List[ChatMessage]) -> List[ChatMessage]:
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
def react_agent(attempt_runtime_improvement: bool = False) -> Agent:
    """
    attempt_runtime_improvement: if True, the agent will attempt to improve the runtime of the previous attempt (always fails for the purposes of the react agent).
    """

    if attempt_runtime_improvement:
        attempts = AgentAttempts(
            attempts=15,
            incorrect_message=incorrect_message,
            score_value=always_fail_metric,
        )
    else:
        attempts = AgentAttempts(
            attempts=5,
            incorrect_message=incorrect_message,
            score_value=pass_metric,
        )

    react_agent = react(
        tools=[
            bash(timeout=300),
            text_editor(timeout=300),
            think(),
            search_tool,
        ],
        attempts=attempts,
        # model=cached_model,  # per‑call caching
        truncation=build_pruner(),
    )
    return react_agent


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
        react_agent(),
    )


# -----------------------------------------------------------------------------
# Task definition --------------------------------------------------------------
# -----------------------------------------------------------------------------


@task
def kernelbench_task(
    level: str = "level_2",
    name: list[str] | None = None,
) -> Task:
    """Return the KernelBench Task object."""

    dataset = hf_dataset(
        "ScalingIntelligence/KernelBench",
        split=level,
        sample_fields=FieldSpec(
            input="code",
            id="problem_id",
            metadata=["name", "level", "code"],
        ),
    )

    # Filter by name if specified
    if name is not None:
        dataset = [sample for sample in dataset if sample.metadata["name"] in name]

    # Verify every dataset sample has timing data
    for sample in dataset:
        level = sample.metadata["level"]
        name = sample.metadata["name"]
        
        # Check if this sample exists in baselines
        matching_baselines = baselines.filter(
            (pl.col("level") == level) & (pl.col("name") == name)
        )
        
        assert matching_baselines.height > 0, (
            f"Dataset sample (level={level}, name={name}) not found in timing baselines. "
            f"Available baselines: {baselines.select('level', 'name').unique().to_pandas().to_string()}"
        )

    return Task(
        dataset=dataset,
        token_limit=500_000,
        solver=kernelbench_solver(),
        sandbox="docker",
        scorer=kernelbench_score(),
    )
