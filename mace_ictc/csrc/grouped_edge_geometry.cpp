#include <torch/extension.h>
torch::Tensor grouped_edge_geometry_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor g,
    torch::Tensor src, torch::Tensor dst, torch::Tensor mask, torch::Tensor v,
    torch::Tensor groups, torch::Tensor columns, int64_t bdim, bool unit,
    int64_t tile_edges, int64_t warps);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward", &grouped_edge_geometry_cuda);
}
