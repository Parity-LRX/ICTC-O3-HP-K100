#include <torch/extension.h>

torch::Tensor csr_message_cuda(
    torch::Tensor x, torch::Tensor b, torch::Tensor w, torch::Tensor g,
    torch::Tensor hx, torch::Tensor hb, torch::Tensor hw, torch::Tensor hg,
    torch::Tensor src, torch::Tensor mask, torch::Tensor ptr,
    torch::Tensor edges, torch::Tensor layout, int64_t channels,
    int64_t output_dim, int64_t paths, bool second, int64_t flags);

torch::Tensor csr_dx_cuda(
    torch::Tensor b, torch::Tensor w, torch::Tensor g, torch::Tensor v,
    torch::Tensor hb, torch::Tensor hw, torch::Tensor hg,
    torch::Tensor dst, torch::Tensor mask, torch::Tensor ptr,
    torch::Tensor edges, torch::Tensor layout, int64_t channels,
    int64_t input_components, bool second, int64_t flags);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("csr_message", &csr_message_cuda,
               "Independent destination-CSR message and cotangent adjoint (CUDA)");
    module.def("csr_dx", &csr_dx_cuda,
               "Independent source-CSR feature adjoint and mixed derivative (CUDA)");
}
