#include <torch/extension.h>
torch::Tensor cached_edge_geometry_cuda(torch::Tensor x,torch::Tensor w,torch::Tensor g,
    torch::Tensor src,torch::Tensor dst,torch::Tensor mask,torch::Tensor v,
    torch::Tensor bm,torch::Tensor cp,bool unit);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("backward",&cached_edge_geometry_cuda);}
