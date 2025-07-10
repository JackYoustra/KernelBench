import pytest
from pathlib import Path

from src.eval import locked_eval_kernel_against_ref

class TestEval:
  @pytest.mark.asyncio
  async def test_eval_kernel_against_typo_in_custom_model(self):
    with open(Path(__file__).parent / "resources" / "sample.py", "r") as o:
      with open(Path(__file__).parent / "resources" / "newsample.py", "r") as f:
        result = await locked_eval_kernel_against_ref(
          original_model_src=o.read(),
          custom_model_src=f.read(),
          seed_num=42,
          num_correct_trials=1,
          num_perf_trials=10,
          verbose=True,
          measure_performance=True,
          build_dir=None,
          gpu_semaphore=None
        )
        
        assert result.model_dump() == {
          "compiled": False,
          "correctness": False,
          "runtime": -1.0,
          "metadata": {
            "compilation_error": "unterminated string literal (detected at line 6) (<string>, line 6)",
            "compile_log": "",
            "device": "0",
            "hardware": "NVIDIA GeForce RTX 5090",
          },
          "runtime_stats": {}
        }
  
  @pytest.mark.asyncio
  async def test_eval_kernel_against_empty_custom_model(self):
    with open(Path(__file__).parent / "resources" / "sample.py", "r") as o:
      result = await locked_eval_kernel_against_ref(
        original_model_src=o.read(),
        custom_model_src="",
        seed_num=42,
        num_correct_trials=1,
        num_perf_trials=10,
        verbose=True,
        measure_performance=True,
        build_dir=None,
        gpu_semaphore=None
      )
      
      assert result.model_dump() == {
        "compiled": False,
        "correctness": False,
        "runtime": -1.0,
        "metadata": {
          "compilation_error": "ModelNew is not defined in the custom model code",
          "compile_log": "",
          "device": "0",
          "hardware": "NVIDIA GeForce RTX 5090",
        },
        "runtime_stats": {}
      }