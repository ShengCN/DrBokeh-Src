#include <torch/extension.h>

#include <vector>
#include <stdio.h>

torch::Tensor scatter_cuda_forward(torch::Tensor x, torch::Tensor lens_effect, torch::Tensor kernel, torch::Tensor lens_mask);
torch::Tensor scatter_cuda_backward(torch::Tensor drgb, torch::Tensor x, torch::Tensor blur, torch::Tensor lens_effect, torch::Tensor kernel, torch::Tensor lens_mask);

std::vector<torch::Tensor> scatter_cuda_no_occlusion_forward(torch::Tensor x, torch::Tensor lens_effect, torch::Tensor kernel, torch::Tensor lens_mask);
torch::Tensor scatter_cuda_no_occlusion_backward(torch::Tensor drgb, torch::Tensor x, torch::Tensor blur, torch::Tensor weights, torch::Tensor lens_effect, torch::Tensor kernel, torch::Tensor lens_mask);

// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/*
 * x:                       B x RGBD x H x W
 * dscatter:                       B x D x H x W
 * lens masked diskernel:   Kernel x Kernel
 *  
*/
torch::Tensor scatter_forward(
    torch::Tensor x,
    torch::Tensor lens_effect,
    torch::Tensor kernel,
    torch::Tensor lens_mask) {
    CHECK_INPUT(x);
    CHECK_INPUT(kernel);
    CHECK_INPUT(lens_mask);

    return scatter_cuda_forward(x, lens_effect, kernel, lens_mask);
}

torch::Tensor scatter_backward(
    torch::Tensor drgb,
    torch::Tensor x,
    torch::Tensor blur,
    torch::Tensor lens_effect,
    torch::Tensor kernel,
    torch::Tensor lens_mask) {

    CHECK_INPUT(drgb);
    CHECK_INPUT(x);
    CHECK_INPUT(kernel);
    CHECK_INPUT(lens_mask);

	return scatter_cuda_backward(drgb, x, blur, lens_effect, kernel, lens_mask);
}


/*
 * x:                       B x RGBD x H x W
 * dscatter:                       B x D x H x W
 * lens masked diskernel:   Kernel x Kernel
 *
*/
std::vector<torch::Tensor> scatter_no_occlusion_forward(
    torch::Tensor x,
    torch::Tensor lens_effect,
    torch::Tensor kernel,
    torch::Tensor lens_mask) {
    CHECK_INPUT(x);
    CHECK_INPUT(kernel);
    CHECK_INPUT(lens_mask);

    return scatter_cuda_no_occlusion_forward(x, lens_effect, kernel, lens_mask);
}

torch::Tensor scatter_no_occlusion_backward(
    torch::Tensor drgb,
    torch::Tensor x,
    torch::Tensor blur,
    torch::Tensor weights,
    torch::Tensor lens_effect,
    torch::Tensor kernel,
    torch::Tensor lens_mask) {

    CHECK_INPUT(drgb);
    CHECK_INPUT(x);
    CHECK_INPUT(kernel);
    CHECK_INPUT(lens_mask);

	return scatter_cuda_no_occlusion_backward(drgb, x, blur, weights, lens_effect, kernel, lens_mask);
}

std::vector<torch::Tensor> multi_scatter_forward(torch::Tensor x, torch::Tensor lens_effect, torch::Tensor kernel, torch::Tensor lens_mask);
torch::Tensor multi_scatter_backward(torch::Tensor drgb, torch::Tensor x, torch::Tensor blur, torch::Tensor weights, torch::Tensor lens_effect, torch::Tensor kernel, torch::Tensor lens_mask);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &scatter_forward, "Scatter forward (CUDA)");
    m.def("backward", &scatter_backward, "Scatter backward (CUDA)");
    m.def("no_occlusion_forward", &scatter_no_occlusion_forward, "Scatter forward (CUDA)");
    m.def("no_occlusion_backward", &scatter_no_occlusion_backward, "Scatter backward (CUDA)");
}
