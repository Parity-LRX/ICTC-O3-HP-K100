#include <torch/extension.h>
void cached_edge_message_cuda(torch::Tensor x,torch::Tensor sh,torch::Tensor w,torch::Tensor g,
    torch::Tensor src,torch::Tensor dst,torch::Tensor mask,torch::Tensor fj,torch::Tensor fa,
    torch::Tensor layout,torch::Tensor output,bool unit_weights);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&cached_edge_message_cuda);}
