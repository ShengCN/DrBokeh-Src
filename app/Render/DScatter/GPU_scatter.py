import torch
import scatter_cuda

class Scatter(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, x, lens_effect, diskernel, lens_mask):
        padding_size = lens_mask.shape[0]//2
        padding_func = torch.nn.ReflectionPad2d(padding_size)
        paddedx = padding_func(x)

        #saved_variables = [paddedx, torch.tensor(lens_effect), diskernel, lens_mask] 
        saved_variables = [x, lens_effect, diskernel, lens_mask]

        blur = scatter_cuda.forward(paddedx, lens_effect, diskernel, lens_mask)
        saved_variables = saved_variables + [blur]
        ctx.save_for_backward(*saved_variables)
        return blur

    @staticmethod
    def backward(ctx, rgb_grad):
        #paddedx, lens_effect, diskernel, lens_mask, blur, weights = ctx.saved_tensors  
        x, lens_effect, diskernel, lens_mask, blur = ctx.saved_tensors

        padding_size = lens_mask.shape[0]//2

        # we use reflect gradient
        padding_func = torch.nn.ReflectionPad2d(padding_size)

        paddedx = padding_func(x)
        padded_blur = padding_func(blur)

        padded_rgb_grad = padding_func(rgb_grad.contiguous())

        grad_x = scatter_cuda.backward(padded_rgb_grad, paddedx, padded_blur, lens_effect, diskernel, lens_mask)

        return grad_x, None, None, None 


class Scatter_no_occlusion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lens_effect, diskernel, lens_mask):
        padding_size = lens_mask.shape[0]//2
        padding_func = torch.nn.ReflectionPad2d(padding_size)
        paddedx      = padding_func(x)

        #saved_variables = [paddedx, torch.tensor(lens_effect), diskernel, lens_mask]
        saved_variables = [x, lens_effect, diskernel, lens_mask]

        blur, weights   = scatter_cuda.no_occlusion_forward(paddedx, lens_effect, diskernel, lens_mask)
        saved_variables = saved_variables + [blur, weights]
        ctx.save_for_backward(*saved_variables)
        return blur

    @staticmethod
    def backward(ctx, rgb_grad):
        #paddedx, lens_effect, diskernel, lens_mask, blur, weights = ctx.saved_tensors
        x, lens_effect, diskernel, lens_mask, blur, weights = ctx.saved_tensors

        padding_size = lens_mask.shape[0]//2

        # we use reflect gradient
        padding_func = torch.nn.ReflectionPad2d(padding_size)

        paddedx = padding_func(x)
        padded_blur = padding_func(blur)
        padded_weight = padding_func(weights)
        padded_rgb_grad = padding_func(rgb_grad.contiguous())

        grad_x = scatter_cuda.no_occlusion_backward(
                padded_rgb_grad,
                paddedx,
                padded_blur,
                padded_weight,
                lens_effect,
                diskernel,
                lens_mask)

        return grad_x, None, None, None
