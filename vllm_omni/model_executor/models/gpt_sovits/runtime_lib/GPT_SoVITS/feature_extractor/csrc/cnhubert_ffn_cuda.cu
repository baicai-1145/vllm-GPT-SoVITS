#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr size_t kWorkspaceBytes = 4 * 1024 * 1024;
constexpr size_t kWorkspaceAlignment = 256;

__global__ void bias_gelu_inplace_kernel(__half* data, const __half* bias, int64_t rows, int64_t cols) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t numel = rows * cols;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        int64_t col = i % cols;
        float x = __half2float(data[i]) + __half2float(bias[col]);
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        data[i] = __float2half(x * cdf);
    }
}

__global__ void bias_inplace_kernel(__half* data, const __half* bias, int64_t rows, int64_t cols) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t numel = rows * cols;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        int64_t col = i % cols;
        data[i] = __hadd(data[i], bias[col]);
    }
}

bool run_linear_row_major(
    cublasHandle_t handle,
    cudaStream_t stream,
    const at::Half* input,
    const at::Half* weight,
    at::Half* output,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
    cublasSetStream(handle, stream);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(out_features),
        static_cast<int>(rows),
        static_cast<int>(in_features),
        &alpha,
        weight,
        CUDA_R_16F,
        static_cast<int>(in_features),
        input,
        CUDA_R_16F,
        static_cast<int>(in_features),
        &beta,
        output,
        CUDA_R_16F,
        static_cast<int>(out_features),
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return status == CUBLAS_STATUS_SUCCESS;
}

void* align_workspace_ptr(void* raw_ptr) {
    auto addr = reinterpret_cast<uintptr_t>(raw_ptr);
    auto aligned = (addr + (kWorkspaceAlignment - 1)) & ~(static_cast<uintptr_t>(kWorkspaceAlignment - 1));
    return reinterpret_cast<void*>(aligned);
}

bool run_lt_matmul_row_major(
    cublasLtHandle_t handle,
    cudaStream_t stream,
    const at::Half* input,
    const at::Half* weight,
    const at::Half* bias,
    at::Half* output,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    cublasLtEpilogue_t epilogue,
    void* workspace_ptr,
    size_t workspace_bytes) {
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatrixLayout_t d_desc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    auto cleanup = [&]() {
        if (preference != nullptr) cublasLtMatmulPreferenceDestroy(preference);
        if (d_desc != nullptr) cublasLtMatrixLayoutDestroy(d_desc);
        if (c_desc != nullptr) cublasLtMatrixLayoutDestroy(c_desc);
        if (b_desc != nullptr) cublasLtMatrixLayoutDestroy(b_desc);
        if (a_desc != nullptr) cublasLtMatrixLayoutDestroy(a_desc);
        if (op_desc != nullptr) cublasLtMatmulDescDestroy(op_desc);
    };

    cublasOperation_t trans_a = CUBLAS_OP_N;
    cublasOperation_t trans_b = CUBLAS_OP_T;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
    cudaDataType_t scale_type = CUDA_R_32F;
    cudaDataType_t data_type = CUDA_R_16F;
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;

    if (cublasLtMatmulDescCreate(&op_desc, compute_type, scale_type) != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return false;
    }
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &data_type, sizeof(data_type));

    if (cublasLtMatrixLayoutCreate(&a_desc, data_type, rows, in_features, in_features) != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return false;
    }
    if (cublasLtMatrixLayoutCreate(&b_desc, data_type, out_features, in_features, in_features) != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return false;
    }
    if (cublasLtMatrixLayoutCreate(&c_desc, data_type, rows, out_features, out_features) != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return false;
    }
    if (cublasLtMatrixLayoutCreate(&d_desc, data_type, rows, out_features, out_features) != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return false;
    }
    cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    cublasLtMatrixLayoutSetAttribute(d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));

    if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return false;
    }
    cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes,
        sizeof(workspace_bytes));

    cublasLtMatmulHeuristicResult_t heuristic {};
    int returned_results = 0;
    auto heuristic_status = cublasLtMatmulAlgoGetHeuristic(
        handle,
        op_desc,
        a_desc,
        b_desc,
        c_desc,
        d_desc,
        preference,
        1,
        &heuristic,
        &returned_results);
    if (heuristic_status != CUBLAS_STATUS_SUCCESS || returned_results <= 0) {
        cleanup();
        return false;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto status = cublasLtMatmul(
        handle,
        op_desc,
        &alpha,
        input,
        a_desc,
        weight,
        b_desc,
        &beta,
        output,
        c_desc,
        output,
        d_desc,
        &heuristic.algo,
        workspace_ptr,
        workspace_bytes,
        stream);
    cleanup();
    return status == CUBLAS_STATUS_SUCCESS;
}

}  // namespace

torch::Tensor cnhubert_ffn_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight1,
    torch::Tensor bias1,
    torch::Tensor weight2,
    torch::Tensor bias2) {
    c10::cuda::CUDAGuard device_guard(input.device());
    auto input_contig = input.contiguous();
    auto weight1_contig = weight1.contiguous();
    auto bias1_contig = bias1.contiguous();
    auto weight2_contig = weight2.contiguous();
    auto bias2_contig = bias2.contiguous();

    const int64_t batch = input_contig.size(0);
    const int64_t seq = input_contig.size(1);
    const int64_t hidden = input_contig.size(2);
    const int64_t inner = weight1_contig.size(0);
    const int64_t out_hidden = weight2_contig.size(0);
    const int64_t rows = batch * seq;

    auto options = input_contig.options().dtype(torch::kFloat16);
    auto input_2d = input_contig.view({rows, hidden});
    auto hidden_2d = torch::empty({rows, inner}, options);
    auto output_2d = torch::empty({rows, out_hidden}, options);
    auto workspace = torch::empty(
        {static_cast<long>(kWorkspaceBytes + kWorkspaceAlignment)},
        input_contig.options().dtype(torch::kUInt8));

    cudaStream_t stream = at::cuda::getDefaultCUDAStream(input.device().index()).stream();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasLtHandle_t lt_handle = reinterpret_cast<cublasLtHandle_t>(handle);
    void* workspace_ptr = align_workspace_ptr(workspace.data_ptr<uint8_t>());

    bool first_ok = run_lt_matmul_row_major(
        lt_handle,
        stream,
        reinterpret_cast<const at::Half*>(input_2d.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(weight1_contig.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(bias1_contig.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(hidden_2d.data_ptr<at::Half>()),
        rows,
        hidden,
        inner,
        CUBLASLT_EPILOGUE_GELU_BIAS,
        workspace_ptr,
        kWorkspaceBytes);
    const int threads = 256;
    if (!first_ok) {
        first_ok = run_linear_row_major(
            handle,
            stream,
            reinterpret_cast<const at::Half*>(input_2d.data_ptr<at::Half>()),
            reinterpret_cast<const at::Half*>(weight1_contig.data_ptr<at::Half>()),
            reinterpret_cast<at::Half*>(hidden_2d.data_ptr<at::Half>()),
            rows,
            hidden,
            inner);
        TORCH_CHECK(first_ok, "cublas first linear failed");
        const int blocks1 = static_cast<int>((hidden_2d.numel() + threads - 1) / threads);
        bias_gelu_inplace_kernel<<<blocks1, threads, 0, stream>>>(
            reinterpret_cast<__half*>(hidden_2d.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(bias1_contig.data_ptr<at::Half>()),
            rows,
            inner);
        AT_CUDA_CHECK(cudaGetLastError());
    }

    bool second_ok = run_lt_matmul_row_major(
        lt_handle,
        stream,
        reinterpret_cast<const at::Half*>(hidden_2d.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(weight2_contig.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(bias2_contig.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(output_2d.data_ptr<at::Half>()),
        rows,
        inner,
        out_hidden,
        CUBLASLT_EPILOGUE_BIAS,
        workspace_ptr,
        kWorkspaceBytes);
    if (!second_ok) {
        second_ok = run_linear_row_major(
            handle,
            stream,
            reinterpret_cast<const at::Half*>(hidden_2d.data_ptr<at::Half>()),
            reinterpret_cast<const at::Half*>(weight2_contig.data_ptr<at::Half>()),
            reinterpret_cast<at::Half*>(output_2d.data_ptr<at::Half>()),
            rows,
            inner,
            out_hidden);
        TORCH_CHECK(second_ok, "cublas second linear failed");
        const int blocks2 = static_cast<int>((output_2d.numel() + threads - 1) / threads);
        bias_inplace_kernel<<<blocks2, threads, 0, stream>>>(
            reinterpret_cast<__half*>(output_2d.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(bias2_contig.data_ptr<at::Half>()),
            rows,
            out_hidden);
        AT_CUDA_CHECK(cudaGetLastError());
    }

    return output_2d.view({batch, seq, out_hidden});
}
