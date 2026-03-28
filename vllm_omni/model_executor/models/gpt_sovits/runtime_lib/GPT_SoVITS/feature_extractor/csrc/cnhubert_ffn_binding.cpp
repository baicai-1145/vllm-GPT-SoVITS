#include <torch/extension.h>

torch::Tensor cnhubert_ffn_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight1,
    torch::Tensor bias1,
    torch::Tensor weight2,
    torch::Tensor bias2);

torch::Tensor cnhubert_ffn_forward(
    torch::Tensor input,
    torch::Tensor weight1,
    torch::Tensor bias1,
    torch::Tensor weight2,
    torch::Tensor bias2) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(weight1.is_cuda(), "weight1 must be CUDA");
    TORCH_CHECK(bias1.is_cuda(), "bias1 must be CUDA");
    TORCH_CHECK(weight2.is_cuda(), "weight2 must be CUDA");
    TORCH_CHECK(bias2.is_cuda(), "bias2 must be CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16, "input must be float16");
    TORCH_CHECK(weight1.scalar_type() == torch::kFloat16, "weight1 must be float16");
    TORCH_CHECK(bias1.scalar_type() == torch::kFloat16, "bias1 must be float16");
    TORCH_CHECK(weight2.scalar_type() == torch::kFloat16, "weight2 must be float16");
    TORCH_CHECK(bias2.scalar_type() == torch::kFloat16, "bias2 must be float16");
    TORCH_CHECK(input.dim() == 3, "input must be [B, T, C]");
    TORCH_CHECK(weight1.dim() == 2 && weight2.dim() == 2, "weights must be rank-2");
    TORCH_CHECK(bias1.dim() == 1 && bias2.dim() == 1, "biases must be rank-1");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight1.is_contiguous(), "weight1 must be contiguous");
    TORCH_CHECK(bias1.is_contiguous(), "bias1 must be contiguous");
    TORCH_CHECK(weight2.is_contiguous(), "weight2 must be contiguous");
    TORCH_CHECK(bias2.is_contiguous(), "bias2 must be contiguous");
    TORCH_CHECK(input.size(2) == weight1.size(1), "input hidden size mismatch");
    TORCH_CHECK(weight1.size(0) == bias1.size(0), "weight1/bias1 mismatch");
    TORCH_CHECK(weight2.size(1) == weight1.size(0), "weight2 input size mismatch");
    TORCH_CHECK(weight2.size(0) == bias2.size(0), "weight2/bias2 mismatch");
    return cnhubert_ffn_forward_cuda(input, weight1, bias1, weight2, bias2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cnhubert_ffn_forward, "CNHuBERT FFN fused forward");
}
