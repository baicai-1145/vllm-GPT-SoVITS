#include <torch/extension.h>

torch::Tensor cnhubert_conv1d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups);

torch::Tensor cnhubert_conv1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16, "input must be float16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "weight must be float16");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [N, C, L]");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D [K, C/groups, R]");
    TORCH_CHECK(stride > 0, "stride must be > 0");
    TORCH_CHECK(dilation > 0, "dilation must be > 0");
    TORCH_CHECK(groups > 0, "groups must be > 0");
    TORCH_CHECK(
        input.size(1) == weight.size(1) * groups,
        "input channels must equal weight.size(1) * groups");
    return cnhubert_conv1d_forward_cuda(
        input.contiguous(),
        weight.contiguous(),
        stride,
        padding,
        dilation,
        groups);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cnhubert_conv1d_forward, "CNHuBERT Conv1d forward");
}
