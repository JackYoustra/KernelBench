import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

tanh_mean_source = R"(
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

torch::Tensor tanh_mean_cuda(torch::Tensor input) {
    CHECK_CUDA(input);
    CHECK_CONTIGUOUS(input);
    if (input.dim() != 4)
        throw std::runtime_error("Input must be 4D tensor");

    int B = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    auto output = torch::empty({B, C, 1, 1}, input.dtype()).to(input.device());

    if (output.numel() == 0) return output;

    int blocks = B * C;
    dim3 threads(256);
    tanh_mean_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                        output.data_ptr<float>(),
                                        B, C, H, W);
    return output;
}

__global__ void tanh_mean_kernel(const float* input, float* output,
                                int B, int C, int H, int W) {
    int in_stride = C * H * W;
    int bc_idx = blockIdx.x;
    
    // Compute batch and channel from block_idx
    int b = bc_idx / C;
    int c = bc_idx % C;
    
    if (b >= B || c >= C) return;
    
    // Compute offset for input
    const float* in = input + b * in_stride + c * H * W;
    
    // Parallel reduction in thread block
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    
    // Compute partial sum
    float sum = 0.0f;
    for (int i = tid; i < H * W; i += 256) {
        sum += in[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Finalize reduction
    for (int s = 128; s >= 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    // Warp reduce for last threads
    if (tid < 32) {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
        if (tid == 0) {
            float mean = sdata[0] / (H * W);
            output[b * C + c] = tanhf(mean);
        }
    }
}

TORCH_LIBRARY_FRAGMENT(model_kernels, m) {
    m.def("tanh_mean_cuda", tanh_mean_cuda);
}
)"

# Compile custom CUDA code
tanh_mean_op = load_inline(
    name="tanh_mean_op",
    cpp_sources=[
        "torch::Tensor tanh_mean_cuda(torch::Tensor input);"
    ],
    cuda_sources=[tanh_mean_source],
    functions=["tanh_mean_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernels for fused tanh(mean(...)) operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                              stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, 
                                  stride=maxpool_stride)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, 
                                  max_val=hardtanh_max)
        self.tanh_mean = tanh_mean_op.tanh_mean_cuda

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        x = self.hardtanh(x)
        x = self.tanh_mean(x)  # Fused operation in custom CUDA kernel
        return x


# Hyperparameters from original model
batch_size = 128
in_channels = 32
out_channels = 64
height, width = 16, 16
kernel_size = 4
stride = 2
padding = 1
maxpool_kernel_size = 2
maxpool_stride = 2
hardtanh_min = -1
hardtanh_max = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, 
            maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max]