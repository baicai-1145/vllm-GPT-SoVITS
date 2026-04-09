#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <array>
#include <limits>
#include <vector>

namespace {

constexpr int kMaxAlgoCount = 8;

inline void check_cudnn(cudnnStatus_t status, const char* message) {
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, message, ": ", cudnnGetErrorString(status));
}

}  // namespace

torch::Tensor cnhubert_conv1d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups) {
    c10::cuda::CUDAGuard device_guard(input.device());

    auto input_4d = input.unsqueeze(2);
    auto weight_4d = weight.unsqueeze(2);

    const auto batch = input_4d.size(0);
    const auto in_channels = input_4d.size(1);
    const auto input_length = input_4d.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(1) * groups == in_channels, "invalid grouped conv shape");

    cudnnHandle_t handle;
    check_cudnn(cudnnCreate(&handle), "cudnnCreate failed");
    cudnnSetStream(handle, c10::cuda::getCurrentCUDAStream().stream());

    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    check_cudnn(cudnnCreateTensorDescriptor(&input_desc), "create input desc failed");
    check_cudnn(cudnnCreateTensorDescriptor(&output_desc), "create output desc failed");
    check_cudnn(cudnnCreateFilterDescriptor(&filter_desc), "create filter desc failed");
    check_cudnn(cudnnCreateConvolutionDescriptor(&conv_desc), "create conv desc failed");

    check_cudnn(
        cudnnSetTensor4dDescriptor(
            input_desc,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_HALF,
            static_cast<int>(batch),
            static_cast<int>(in_channels),
            1,
            static_cast<int>(input_length)),
        "set input desc failed");

    check_cudnn(
        cudnnSetFilter4dDescriptor(
            filter_desc,
            CUDNN_DATA_HALF,
            CUDNN_TENSOR_NCHW,
            static_cast<int>(out_channels),
            static_cast<int>(weight.size(1)),
            1,
            static_cast<int>(kernel_size)),
        "set filter desc failed");

    const std::array<int, 2> pad_dims = {0, static_cast<int>(padding)};
    const std::array<int, 2> stride_dims = {1, static_cast<int>(stride)};
    const std::array<int, 2> dilation_dims = {1, static_cast<int>(dilation)};
    check_cudnn(
        cudnnSetConvolutionAndDescriptor(
            conv_desc,
            static_cast<int>(pad_dims.size()),
            pad_dims.data(),
            stride_dims.data(),
            dilation_dims.data(),
            CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_FLOAT),
        "set conv desc failed");
    check_cudnn(cudnnSetConvolutionGroupCount(conv_desc, static_cast<int>(groups)), "set group count failed");
    check_cudnn(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION), "set math type failed");

    const int64_t output_w =
        ((input_length + 2 * padding - (dilation * (kernel_size - 1)) - 1) / stride) + 1;
    TORCH_CHECK(output_w > 0, "invalid conv output width");
    const int output_n = static_cast<int>(batch);
    const int output_c = static_cast<int>(out_channels);
    const int output_h = 1;

    auto output_4d = torch::empty(
        {output_n, output_c, output_h, output_w},
        input.options());

    const int64_t elements_per_batch = in_channels * input_length;
    int64_t max_batch_chunk = 4;
    if (elements_per_batch <= 0) {
        max_batch_chunk = 1;
    }

    cudnnConvolutionFwdAlgo_t selected_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    size_t max_workspace_bytes = 0;
    std::vector<int64_t> chunk_sizes;
    for (int64_t offset = 0; offset < batch; offset += max_batch_chunk) {
        chunk_sizes.push_back(std::min<int64_t>(max_batch_chunk, batch - offset));
    }

    for (size_t chunk_index = 0; chunk_index < chunk_sizes.size(); ++chunk_index) {
        const int current_batch = static_cast<int>(chunk_sizes[chunk_index]);
        check_cudnn(
            cudnnSetTensor4dDescriptor(
                input_desc,
                CUDNN_TENSOR_NCHW,
                CUDNN_DATA_HALF,
                current_batch,
                static_cast<int>(in_channels),
                1,
                static_cast<int>(input_length)),
            "set input desc failed");
        check_cudnn(
            cudnnSetTensor4dDescriptor(
                output_desc,
                CUDNN_TENSOR_NCHW,
                CUDNN_DATA_HALF,
                current_batch,
                output_c,
                output_h,
                static_cast<int>(output_w)),
            "set output desc failed");

        int returned_algo_count = 0;
        cudnnConvolutionFwdAlgoPerf_t perf_results[kMaxAlgoCount];
        check_cudnn(
            cudnnGetConvolutionForwardAlgorithm_v7(
                handle,
                input_desc,
                filter_desc,
                conv_desc,
                output_desc,
                kMaxAlgoCount,
                &returned_algo_count,
                perf_results),
            "get forward algo failed");
        TORCH_CHECK(returned_algo_count > 0, "no cudnn forward algo available");

        cudnnConvolutionFwdAlgo_t chunk_algo = perf_results[0].algo;
        for (int index = 0; index < returned_algo_count; ++index) {
            if (perf_results[index].status == CUDNN_STATUS_SUCCESS) {
                chunk_algo = perf_results[index].algo;
                break;
            }
        }
        selected_algo = chunk_algo;

        size_t workspace_bytes = 0;
        check_cudnn(
            cudnnGetConvolutionForwardWorkspaceSize(
                handle,
                input_desc,
                filter_desc,
                conv_desc,
                output_desc,
                selected_algo,
                &workspace_bytes),
            "get workspace size failed");
        if (workspace_bytes > max_workspace_bytes) {
            max_workspace_bytes = workspace_bytes;
        }
    }

    torch::Tensor workspace;
    void* workspace_ptr = nullptr;
    if (max_workspace_bytes > 0) {
        workspace = torch::empty(
            {static_cast<long long>(max_workspace_bytes)},
            torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
        workspace_ptr = workspace.data_ptr();
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    int64_t batch_offset = 0;
    for (size_t chunk_index = 0; chunk_index < chunk_sizes.size(); ++chunk_index) {
        const int current_batch = static_cast<int>(chunk_sizes[chunk_index]);
        auto input_chunk = input_4d.narrow(0, batch_offset, current_batch);
        auto output_chunk = output_4d.narrow(0, batch_offset, current_batch);

        check_cudnn(
            cudnnSetTensor4dDescriptor(
                input_desc,
                CUDNN_TENSOR_NCHW,
                CUDNN_DATA_HALF,
                current_batch,
                static_cast<int>(in_channels),
                1,
                static_cast<int>(input_length)),
            "set input desc failed");
        check_cudnn(
            cudnnSetTensor4dDescriptor(
                output_desc,
                CUDNN_TENSOR_NCHW,
                CUDNN_DATA_HALF,
                current_batch,
                output_c,
                output_h,
                static_cast<int>(output_w)),
            "set output desc failed");

        int returned_algo_count = 0;
        cudnnConvolutionFwdAlgoPerf_t perf_results[kMaxAlgoCount];
        check_cudnn(
            cudnnGetConvolutionForwardAlgorithm_v7(
                handle,
                input_desc,
                filter_desc,
                conv_desc,
                output_desc,
                kMaxAlgoCount,
                &returned_algo_count,
                perf_results),
            "get forward algo failed");
        TORCH_CHECK(returned_algo_count > 0, "no cudnn forward algo available");

        cudnnConvolutionFwdAlgo_t chunk_algo = perf_results[0].algo;
        for (int index = 0; index < returned_algo_count; ++index) {
            if (perf_results[index].status == CUDNN_STATUS_SUCCESS) {
                chunk_algo = perf_results[index].algo;
                break;
            }
        }

        size_t workspace_bytes = 0;
        check_cudnn(
            cudnnGetConvolutionForwardWorkspaceSize(
                handle,
                input_desc,
                filter_desc,
                conv_desc,
                output_desc,
                chunk_algo,
                &workspace_bytes),
            "get workspace size failed");

        check_cudnn(
            cudnnConvolutionForward(
                handle,
                &alpha,
                input_desc,
                input_chunk.data_ptr(),
                filter_desc,
                weight_4d.data_ptr(),
                conv_desc,
                chunk_algo,
                workspace_ptr,
                workspace_bytes,
                &beta,
                output_desc,
                output_chunk.data_ptr()),
            "cudnnConvolutionForward failed");
        batch_offset += current_batch;
    }

    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroy(handle);
    return output_4d.squeeze(2);
}
