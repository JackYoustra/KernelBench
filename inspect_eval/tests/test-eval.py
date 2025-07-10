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
  
  @pytest.mark.asyncio
  async def test_eval_kernel_against_compile_error(self):
    with open(Path(__file__).parent / "resources" / "sample.py", "r") as o:
      with open(Path(__file__).parent / "resources" / "compile_error.py", "r") as f:
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
            "compilation_error": "Error building extension 'model_kernels'",
            "compile_log": R"""[1/3] c++ -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=model_kernels -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1016\" -isystem /home/jack/code/llm/KernelBench/.venv/lib/python3.12/site-packages/torch/include -isystem /home/jack/code/llm/KernelBench/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/local/cuda/include -isystem /usr/include/python3.12 -D_GLIBCXX_USE_CXX11_ABI=1 -fPIC -std=c++17 -O3 -arch=sm_120 -c /home/jack/.cache/torch_extensions/py312_cu128/model_kernels/main.cpp -o main.o 
FAILED: main.o 
c++ -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=model_kernels -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1016\" -isystem /home/jack/code/llm/KernelBench/.venv/lib/python3.12/site-packages/torch/include -isystem /home/jack/code/llm/KernelBench/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/local/cuda/include -isystem /usr/include/python3.12 -D_GLIBCXX_USE_CXX11_ABI=1 -fPIC -std=c++17 -O3 -arch=sm_120 -c /home/jack/.cache/torch_extensions/py312_cu128/model_kernels/main.cpp -o main.o 
c++: error: unrecognized command-line option ‘-arch=sm_120’
[2/3] /usr/local/cuda/bin/nvcc --generate-dependencies-with-compile --dependency-output cuda.cuda.o.d -DTORCH_EXTENSION_NAME=model_kernels -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1016\" -isystem /home/jack/code/llm/KernelBench/.venv/lib/python3.12/site-packages/torch/include -isystem /home/jack/code/llm/KernelBench/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/local/cuda/include -isystem /usr/include/python3.12 -D_GLIBCXX_USE_CXX11_ABI=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_120,code=compute_120 -gencode=arch=compute_120,code=sm_120 --compiler-options '-fPIC' -std=c++17 -c /home/jack/.cache/torch_extensions/py312_cu128/model_kernels/cuda.cu -o cuda.cuda.o 
FAILED: cuda.cuda.o 
/usr/local/cuda/bin/nvcc --generate-dependencies-with-compile --dependency-output cuda.cuda.o.d -DTORCH_EXTENSION_NAME=model_kernels -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1016\" -isystem /home/jack/code/llm/KernelBench/.venv/lib/python3.12/site-packages/torch/include -isystem /home/jack/code/llm/KernelBench/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/local/cuda/include -isystem /usr/include/python3.12 -D_GLIBCXX_USE_CXX11_ABI=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_120,code=compute_120 -gencode=arch=compute_120,code=sm_120 --compiler-options '-fPIC' -std=c++17 -c /home/jack/.cache/torch_extensions/py312_cu128/model_kernels/cuda.cu -o cuda.cuda.o 
/home/jack/.cache/torch_extensions/py312_cu128/model_kernels/cuda.cu(78): error: identifier "input_width" is undefined
                                                 ih * input_width +
                                                      ^

1 error detected in the compilation of "/home/jack/.cache/torch_extensions/py312_cu128/model_kernels/cuda.cu".
ninja: build stopped: subcommand failed.
""",
            "device": "0",
            "hardware": "NVIDIA GeForce RTX 5090",
          },
          "runtime_stats": {}
        }
      
