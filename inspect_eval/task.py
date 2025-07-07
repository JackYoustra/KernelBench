from textwrap import dedent
import asyncio
from enum import Enum
from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.solver import generate, solver, system_message, use_tools, TaskState, Solver, Generate
from inspect_ai.scorer import Score, Target, Scorer
from inspect_ai.tool import bash, text_editor, think, web_search
from inspect_ai.util import sandbox

from src.eval import locked_eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from src.utils import extract_first_code, set_gpu_arch


# if it's correct
# return the time it took to run
class KernelBenchScoreType(Enum):
    NOT_EXIST = "not_exist"
    NOT_EXTRACTED = "not_extracted"
    NOT_COMPILES = "not_compiles"
    NOT_CORRECT = "not_correct"

def kernelbench_score() -> Scorer:

  # we only want one kernel to run at a time
  gpu_semaphore = asyncio.Semaphore(1)

  async def kernelbench_score(state: TaskState, target: Target) -> Score:
    # get current submit_kernel in docker container
    # grade
    # value for exist, compiles, correct, and runtime

    # Example usage:
    # return exist_score()
    # return compiles_score() 
    # return correct_score(1.234)

    try:
        contents = await sandbox().read_file("model_new.py")
    except FileNotFoundError:
        return Score(value=KernelBenchScoreType.NOT_EXIST.value)
    
    custom_cuda = extract_first_code(contents, ["python", "cpp"])

    if custom_cuda is None:
        return Score(value=KernelBenchScoreType.NOT_EXTRACTED.value)

    # TODO: get ref_arch_src
    if isinstance(state.input[0], str):
      ref_arch_src = state.input[0]
    else:
      ref_arch_src = state.input[0].text

    kernel_exec_result = await locked_eval_kernel_against_ref(
        ref_arch_src,
        custom_cuda,
        verbose=True,
        measure_performance=True,
        num_correct_trials=5,
        num_perf_trials=100,
        gpu_semaphore=gpu_semaphore
    )

    if not kernel_exec_result.compiled:
        return Score(value=KernelBenchScoreType.NOT_COMPILES.value)
    
    if not kernel_exec_result.correctness:
        return Score(value=KernelBenchScoreType.NOT_CORRECT.value)

    return Score(value=kernel_exec_result.runtime)
  
  return kernelbench_score

@solver
def cudaify_prompt() -> Solver:
    """Transform the prompt template.

    Prompt template containing a transform function and any
    number of additional `params`. All values contained in sample
    `metadata` and `store` are also automatically included in the
    `params`.

    Args:
      transform: Transform function.
      **params: Parameters to fill into the template.

    Returns:
      A solver that uses the specified prompt template.
    """

    problem_instruction = dedent("""
      Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in a file called model_new.py in the sandbox that your bash tool has access to.
    """)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        set_gpu_arch(["Blackwell"])
        prompt = state.user_prompt
        prompt.text = prompt_generate_custom_cuda_from_prompt_template(prompt.text, problem_instruction=problem_instruction)
        return state

    return solve

@solver
def kernelbench_solver():
    # normal_system_message = system_message(prompt_generate_custom_cuda())
    # we can put this in setup, but if we put it here
    # it'd be substituted
    # if a different solver is applied
    # doing it this way for now
    normal_prompt = cudaify_prompt()

    normal_system_message = system_message(dedent("""
      You are a world-class expert in CUDA programming.
      You are developing GPU kernels for a 5090 (sm_120) released in Feb 1, 2025.
      Your CUDA version is 12.9.
                                                  
      You have access to various tools to help you with your task.
      For the bash tool, there is a folder called materials that contains a manual for the 5090 and its requisite operations in markdown. Reference it to help you. You also have access to the internet if need be, and can use wget or curl in conjunction with your internet searches.
    """))

    think_system_message = system_message(dedent("""
      Use the think tool to think about something. It will not obtain
      new information or make any changes to the repository, but just 
      log the thought. Use it when complex reasoning or brainstorming
      is needed. For example, if you explore the repo and discover
      the source of a bug, call this tool to brainstorm several unique
      ways of fixing the bug, and assess which change(s) are likely to 
      be simplest and most effective. Alternatively, if you receive
      some test results, call this tool to brainstorm ways to fix the
      failing tests.
  """))

    return react(
      [
        normal_system_message,
        think_system_message,
        normal_prompt,
        use_tools([
          bash(timeout=300),
          text_editor(timeout=300),
          think(),
          web_search(["openai", "anthropic", "gemini", "perplexity", "exa"])
        ]),
        generate(),
      ],
      attempts=5
    )

@task
def kernelbench_task():
    dataset = hf_dataset(
        "ScalingIntelligence/KernelBench",
        split="level_2",
        field_specs=[
            FieldSpec(name="code", type=str),
            FieldSpec(name="name", type=str),
            FieldSpec(name="problem_id", type=int),
            FieldSpec(name="level", type=int),
        ]
    )

    for sample in dataset:
        assert sample.files is None
        sample.files = {
            "materials": "materials"
        }

    return Task(
        dataset=dataset,
        solver=kernelbench_solver(),
        sandbox="docker",
        scorer=kernelbench_score,
    )