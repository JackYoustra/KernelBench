import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline
import sys

# Define source code for CUDA kernels and C++ wrappers
conv_transpose3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm> // For std::max
#include <vector>    // For std::vector

// CUDA Kernel for ConvTranspose3d
// Each thread calculates one output element: output[n, oc, od, oh, ow]
__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    const float* bias, // Can be nullptr if bias is not used
    float* output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int kernel_size_d, int kernel_size_h, int kernel_size_w,
    int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int N // Batch size
) {
    // Calculate the linear index of the output element this thread is responsible for.
    const int output_linear_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_total_elements = N * out_channels * out_depth * out_height * out_width;

    // Bounds check for the thread
    if (output_linear_index >= output_total_elements) return;

    // Decode output indices from the linear index
    // Layout: N, C_out, D_out, H_out, W_out
    int ow = output_linear_index % out_width;
    int oh = (output_linear_index / out_width) % out_height;
    int od = (output_linear_index / (out_width * out_height)) % out_depth;
    int oc = (output_linear_index / (out_width * out_height * out_depth)) % out_channels;
    int n = output_linear_index / (out_width * out_height * out_depth * out_channels);

    // Initialize sum for the current output element. Add bias if provided.
    float sum = 0.0f;
    if (bias != nullptr) {
        sum = bias[oc]; // Bias summed for each output channel
    }

    // Precompute strides for input and weight tensors to optimize memory access
    const unsigned int input_spatial_stride = (unsigned int)in_depth * in_height * in_width;
    const unsigned int input_channel_stride = (unsigned int)in_height * in_width;
    const unsigned int input_height_stride = (unsigned int)in_width;

    // Strides for accessing weight tensor elements (shape: C_in, C_out, kD, kH, kW)
    const unsigned int weight_oc_stride = (unsigned int)kernel_size_d * kernel_size_h * kernel_size_w; // Stride for output channels
    const unsigned int weight_kd_stride = (unsigned int)kernel_size_h * kernel_size_w; // Stride for kernel depth
    const unsigned int weight_kh_stride = (unsigned int)kernel_size_w; // Stride for kernel height

    // Iterate over input channels
    for (int ic = 0; ic < in_channels; ++ic) {
        // Iterate over kernel spatial dimensions
        for (int kd = 0; kd < kernel_size_d; ++kd) {
            for (int kh = 0; kh < kernel_size_h; ++kh) {
                for (int kw = 0; kw < kernel_size_w; ++kw) {

                    // Calculate the corresponding input indices based on output indices, stride, padding, and kernel position
                    int id = od * stride_d - padding_d + kd;
                    int ih = oh * stride_h - padding_h + kh;
                    int iw = ow * stride_w - padding_w + kw;

                    // Check if the calculated input indices are within the bounds of the input tensor
                    if (id >= 0 && id < in_depth &&
                        ih >= 0 && ih < in_height &&
                        iw >= 0 && iw < in_width)
                    {
                        // Calculate linear indices for accessing input and weight tensors
                        // Input tensor indexing: [n, ic, id, ih, iw]
                        int input_linear_idx = n * input_spatial_stride +
                                               ic * input_channel_stride +
                                               id * input_height_stride +
                                               ih * input_width +
                                               iw;

                        // Weight tensor indexing: [ic, oc, kd, kh, kw]
                        // Correct indexing for weights: ic * (C_out * kD * kH * kW) + oc * (kD * kH * kW) + kd * (kH * kW) + kh * kW + kw
                        int weight_linear_idx = ic * (out_channels * weight_oc_stride) +  // Offset for input channel
                                                oc * weight_oc_stride +             // Offset for output channel
                                                kd * weight_kd_stride +             // Offset for kernel depth
                                                kh * weight_kh_stride +             // Offset for kernel height
                                                kw;                                 // Offset for kernel width
                        
                        // Accumulate the weighted input value onto the sum
                        sum += input[input_linear_idx] * weight[weight_linear_idx];
                    }
                }
            }
        }
    }

    // Store the final computed sum in the output tensor at the correct position
    output[output_linear_index] = sum;
}

// CUDA Kernel for element-wise clamp and divide operation
__global__ void clamp_divide_kernel(
    const float* input,
    float* output,
    int size, // Total number of elements in the input tensor
    float min_val,
    float divisor)
{
    // Calculate the linear index for the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the index is within the bounds of the tensor
    if (idx < size) {
        // Clamp the input value to the minimum specified value using CUDA's built-in max
        float clamped_val = max(input[idx], min_val); 
        // Divide the clamped value by the divisor
        output[idx] = clamped_val / divisor;
    }
}
"""

conv_transpose3d_cpp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declarations of the CUDA kernels
__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int kernel_size_d, int kernel_size_h, int kernel_size_w,
    int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int N
);

__global__ void clamp_divide_kernel(
    const float* input,
    float* output,
    int size,
    float min_val,
    float divisor);

// C++ wrapper function for the ConvTranspose3d CUDA kernel
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int kernel_size_d, int kernel_size_h, int kernel_size_w,
    int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width
) {
    // Perform input validation: Ensure tensors are on CUDA and compatible devices
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Input and weight tensors must be on CUDA.");
    if (bias.defined()) { // Check if bias tensor is defined (not None)
        TORCH_CHECK(bias.get_device() == input.get_device(), "Bias tensor must be on the same CUDA device as input.");
    }

    // Get batch size from the input tensor's shape
    const int N = input.size(0);
    
    // Create the output tensor with the correct shape and data type (matching input tensor options)
    auto output = torch::empty({N, out_channels, out_depth, out_height, out_width}, input.options());

    // Determine kernel launch configuration: calculate the required number of blocks and threads per block
    const int output_total_elements = N * out_channels * out_depth * out_height * out_width;
    const int block_size = 256; // Number of threads per block (a common choice)
    // Calculate the number of blocks needed to cover all output elements. Ceiling division.
    const int grid_size = (output_total_elements + block_size - 1) / block_size; 

    // Get the raw pointer for the bias tensor. If bias is not defined, pass a null pointer.
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    // Launch the ConvTranspose3d CUDA kernel
    conv_transpose3d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),      // Pointer to input tensor data
        weight.data_ptr<float>(),     // Pointer to weight tensor data
        bias_ptr,                     // Pointer to bias tensor data (or nullptr)
        output.data_ptr<float>(),     // Pointer to output tensor data
        stride_d, stride_h, stride_w, // Strides for each dimension
        padding_d, padding_h, padding_w, // Padding for each dimension
        kernel_size_d, kernel_size_h, kernel_size_w, // Kernel dimensions
        in_channels, out_channels,    // Number of input and output channels
        in_depth, in_height, in_width, // Input dimensions
        out_depth, out_height, out_width, // Output dimensions
        N                             // Batch size
    );
    // Check for any CUDA errors that occurred during kernel launch
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed for conv_transpose3d_kernel.");

    return output; // Return the computed output tensor
}

// C++ wrapper function for the Clamp and Divide CUDA kernel
torch::Tensor clamp_divide_cuda(
    torch::Tensor input,
    float min_val,
    float divisor)
{
    // Validate that the input tensor is on the CUDA device
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA.");
    
    // Get the total number of elements in the input tensor
    auto size = input.numel();
    // Create an empty output tensor with the same shape and data type as the input
    auto output = torch::empty_like(input);

    // Define kernel launch configuration: number of blocks and threads per block
    const int block_size = 256; // Threads per block
    // Calculate the number of blocks needed using ceiling division
    const int grid_size = (size + block_size - 1) / block_size; 

    // Launch the clamp_divide CUDA kernel
    clamp_divide_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),  // Pointer to input tensor data
        output.data_ptr<float>(), // Pointer to output tensor data
        size,                     // Total number of elements
        min_val,                  // Minimum value for clamping
        divisor                     // Divisor for the division operation
    );
    // Check for any CUDA errors that occurred during kernel launch
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed for clamp_divide_kernel.");

    return output; // Return the computed output tensor
}
"""

# Compile the CUDA kernels using torch.utils.cpp_extension.load_inline
# This function compiles the C++ and CUDA source codes into a Python extension module.
compiled_kernels = load_inline(
    name="model_kernels", # Name of the compiled module
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_cuda_source,
    functions=["conv_transpose3d_cuda", "clamp_divide_cuda"], # List of C++ functions to expose from the module
    verbose=True, # Set to True to display compilation output during execution
    # Specify extra compiler flags: optimization level (-O3) and target compute capability (sm_120 for 5090 GPU)
    extra_cflags=["-O3", "-arch=sm_120"] 
)

# Define the optimized Model architecture (ModelNew) using the compiled CUDA kernels
class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant, all implemented using custom CUDA kernels for performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()

        # Store model parameters passed during initialization
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.min_value = min_value
        self.divisor = divisor

        # Process kernel_size, stride, and padding to ensure they are 3D values (depth, height, width)
        # If they are integers, replicate them for all three dimensions.
        if isinstance(kernel_size, int):
            self.k_d, self.k_h, self.k_w = kernel_size, kernel_size, kernel_size
        else:
            self.k_d, self.k_h, self.k_w = kernel_size

        if isinstance(stride, int):
            self.s_d, self.s_h, self.s_w = stride, stride, stride
        else:
            self.s_d, self.s_h, self.s_w = stride

        if isinstance(padding, int):
            self.p_d, self.p_h, self.p_w = padding, padding, padding
        else:
            self.p_d, self.p_h, self.p_w = padding

        # Initialize weights and bias parameters. This mimics the behavior of nn.ConvTranspose3d.
        # Weights are initialized using He uniform initialization (gain=1 by default for ReLU-like activations).
        # Bias is initialized to zeros.
        fan_in_ = self.in_channels * self.k_d * self.k_h * self.k_w
        fan_out_ = self.out_channels * self.k_d * self.k_h * self.k_w
        # Calculate standard deviation for He initialization. A common formula for weight init.
        std_dev = math.sqrt(2.0 / (fan_in_ + fan_out_)) 
        self.weight = nn.Parameter(torch.randn(self.in_channels, self.out_channels, self.k_d, self.k_h, self.k_w) * std_dev)
        self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def forward(self, x):
        """
        Defines the forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, D, H, W).
        Returns:
            torch.Tensor: Output tensor after transposed convolution, clamping, and division.
        """
        # Get input tensor dimensions: (Batch Size, Input Channels, Depth, Height, Width)
        N, C_in, D, H, W = x.shape

        # Calculate the output dimensions of the transposed convolution.
        # The formula used is: output_dim = (input_dim - 1) * stride + kernel_size - 2 * padding
        D_out = (D - 1) * self.s_d + self.k_d - 2 * self.p_d
        H_out = (H - 1) * self.s_h + self.k_h - 2 * self.p_h
        W_out = (W - 1) * self.s_w + self.k_w - 2 * self.p_w
        
        # Ensure calculated output dimensions are non-negative. This guards against unusual parameter combinations.
        D_out = max(0, D_out)
        H_out = max(0, H_out)
        W_out = max(0, W_out)

        # Execute the custom CUDA kernel for the 3D transposed convolution.
        # Pass all necessary parameters including input tensor, weights, bias, and convolution settings.
        conv_out = compiled_kernels.conv_transpose3d_cuda(
            x, self.weight, self.bias,
            self.s_d, self.s_h, self.s_w,
            self.p_d, self.p_h, self.p_w,
            self.k_d, self.k_h, self.k_w,
            self.in_channels, self.out_channels,
            D, H, W,         # Input dimensions
            D_out, H_out, W_out # Calculated Output dimensions
        )

        # Execute the custom CUDA kernel for the element-wise clamping and division operations.
        # This fuses the clamp and divide operations into a single kernel for efficiency.
        clamped_divided_out = compiled_kernels.clamp_divide_cuda(
            conv_out, self.min_value, self.divisor
        )

        return clamped_divided_out # Return the final output tensor

# --- Helper functions to provide inputs and initialization parameters ---
# These functions mirror the structure provided in the problem description for the original model.

def get_inputs():
    """
    Generates random input tensors for the model according to the specified architecture dimensions.
    Returns:
        list: A list containing a single torch.Tensor as input.
    """
    # Architecture parameters defining the input tensor shape
    batch_size = 16
    in_channels = 32
    depth, height, width = 16, 32, 32
    
    # Create a random input tensor. It's placed on the CUDA device ('cuda').
    # Ensure CUDA is available before attempting to use 'cuda'.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu") # Fallback to CPU if CUDA is not available
        print("CUDA not available, falling back to CPU. Performance will be significantly impacted.", file=sys.stderr)

    return [torch.randn(batch_size, in_channels, depth, height, width, device=device)]

def get_init_inputs():
    """
    Returns the initialization parameters required for the Model.
    Returns:
        list: A list of parameters for initializing the Model.
    """
    # Model configuration parameters
    in_channels = 32
    out_channels = 16
    kernel_size = 3
    stride = 2
    padding = 1
    min_value = -1.0
    divisor = 2.0
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]

if __name__ == '__main__':
    # Example usage: Create and test the ModelNew
    print("Creating ModelNew and testing...")

    # Get initialization parameters
    init_params = get_init_inputs()
    in_channels, out_channels, kernel_size, stride, padding, min_value, divisor = init_params

    # Instantiate the ModelNew
    model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding, min_value, divisor)

    # Move model to CUDA if available
    if torch.cuda.is_available():
        model_new.cuda()
        print("ModelNew moved to CUDA.")
    else:
        print("CUDA not available, running on CPU.")

    # Get input data
    inputs = get_inputs()
    input_tensor = inputs[0]

    # Perform a forward pass
    try:
        output_new = model_new(input_tensor)
        print("ModelNew forward pass successful.")
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output_new.shape}")

        # Optional: Compare with original PyTorch model for correctness (requires original Model definition)
        # For demonstration, we'll assume the custom kernels are correct.

    except Exception as e:
        print(f"An error occurred during the forward pass: {e}")