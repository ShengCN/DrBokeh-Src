#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
    __device__ int is_clip_grad = 1;

    template <typename scalar_t>
    __device__ __forceinline__ scalar_t sign(scalar_t a) {
        if (a > 0.0) {
            return 1.0;
        }
        else {
            return -1.0;
        }
    }

    template <typename scalar_t>
    __device__ __forceinline__ scalar_t clip_grad(scalar_t a) {
        if (is_clip_grad) {
            if (abs(a) >= 1e-2f) {
                return a;
            }

            if (abs(a) < 1e-2f && abs(a) >= 1e-4f)
                return max(abs(a), 1e-2f) * sign(a);

            if (abs(a) < 1e-4f && abs(a) >= 1e-6f)
                return max(abs(a), 1e-3f) * sign(a);

            return a;
        }

        return a;
    }

    template <typename scalar_t>
    __device__ __forceinline__ scalar_t myexp(scalar_t a) {
        return expf(a);
        // return expf(a);
        // return __expf(a);
    }

    template <typename scalar_t> __device__ __forceinline__ scalar_t softstep(scalar_t a, scalar_t b) {
        float eps = 1e-6f;
        /* if a > b 1.0 else 0.0 */
        scalar_t s = 0.1, g = 3.0, d = 0.0;
        // scalar_t s = 1, g = 5.0, d = 0.0;
        return s/(s + myexp(-(a-b+d)*g) + eps);
    }

    template <typename scalar_t>
    __device__ __forceinline__ scalar_t d_softstep(scalar_t a, scalar_t b) {
        float eps = 1e-6f;
        scalar_t s = 0.1, g = 3.0, d = 0.0;
        // scalar_t s = 1, g = 5.0, d = 0.0;
        scalar_t exp_term = myexp(g * (-(a-b+d)));
        return (g * s * exp_term)/((exp_term + s) * (exp_term + s) + eps);
    }


    template <typename scalar_t>
    __device__ __forceinline__ scalar_t sigmoid(scalar_t a) {
        scalar_t neg = myexp(-a);

        return 1.0/(1.0+neg);
    }


    template <typename scalar_t>
    __device__ __forceinline__ scalar_t tanh(scalar_t a) {
        float eps = 1e-6f;
        scalar_t pos = myexp(a);
        scalar_t neg = myexp(-a);

        return (pos - neg)/(pos + neg + eps);
    }


    template <typename scalar_t>
    __device__ __forceinline__ scalar_t d_tanh(scalar_t a) {
        // return 1.0-(tanh(a) * tanh(a)); // bug
        float eps = 1e-6f;
        scalar_t div = myexp(a) + myexp(-a);

        return 4.0/(div * div + eps);
    }


    template <typename scalar_t>
    __device__ __forceinline__ scalar_t soft_occlusion(scalar_t center_disp,
                                                       scalar_t neighbour_disp,
                                                       scalar_t lens_effect,
                                                       bool is_center) {
        // return center_disp;
        // return neighbour_disp;

        if (is_center) { /* Ignore the occlusion for center pixel */
            return 1.0;
        }

        scalar_t rel_dis     = neighbour_disp - center_disp;
        scalar_t scatter_dis = abs(center_disp) * lens_effect;

        scalar_t in_focal = (1.0-myexp(-scatter_dis * scatter_dis * 3.0));
        scalar_t rel_occ  = 0.5 + 0.5 * tanh(10.0 * (rel_dis-0.1));

        return (1.0-in_focal) * rel_occ + in_focal;
    }


    template <typename scalar_t>
    __device__ __forceinline__ scalar_t d_soft_occlusion_nd(scalar_t center_disp,
                                                            scalar_t neighbour_disp,
                                                            scalar_t lens_effect,
                                                            bool is_center) {
        if (is_center) {
            return 0.0;
        }

        scalar_t rel_dis = neighbour_disp - center_disp;
        scalar_t scatter_dis = abs(center_disp) * lens_effect;

        scalar_t in_focal = (1.0-myexp(-scatter_dis * scatter_dis * 3.0));
        scalar_t rel_occ  = 0.5 + 0.5 * tanh(10.0 * (rel_dis-0.1));

        /*
         * in_focal(cd)
         * rel_occ(nd, cd)
         * */
        scalar_t drel_dis = 1.0;
        scalar_t d_rel_occ  = 0.5 * d_tanh(10.0 * (rel_dis-0.1)) * 10.0 * drel_dis;

        d_rel_occ = clip_grad(d_rel_occ);
        return (1.0-in_focal) * d_rel_occ;
    }


    template <typename scalar_t>
    __device__ __forceinline__ scalar_t d_soft_occlusion_cd(scalar_t center_disp,
                                                            scalar_t neighbour_disp,
                                                            scalar_t lens_effect,
                                                            bool is_center) {
        if (is_center) {
            return 0.0;
        }

        scalar_t rel_dis = neighbour_disp - center_disp;
        scalar_t scatter_dis = abs(center_disp) * lens_effect;

        scalar_t in_focal = (1.0-myexp(-scatter_dis * scatter_dis * 3.0));
        scalar_t rel_occ  = 0.5 + 0.5 * tanh(10.0 * (rel_dis-0.1));

        /*
         * in_focal(cd)
         * rel_occ(nd, cd)
         * */

        scalar_t drel_dis = -1.0;
        scalar_t dscatter_dis = lens_effect * sign(center_disp);

        scalar_t d_in_focal = -myexp(-3.0 * scatter_dis * scatter_dis) * (-6.0 * scatter_dis) * dscatter_dis;
        scalar_t d_rel_occ  = 0.5 * d_tanh(10.0 * (rel_dis-0.1)) * 10.0 * drel_dis;


        d_in_focal = clip_grad(d_in_focal);
        d_rel_occ  = clip_grad(d_rel_occ);

        return d_in_focal + (1.0-in_focal) * d_rel_occ - d_in_focal * rel_occ;
    }


    template <typename scalar_t>
    __device__ __forceinline__ scalar_t d_soft_occlusion(scalar_t center_disp,
                                                         scalar_t neighbour_disp,
                                                         scalar_t lens_effect,
                                                         bool is_center) {
        /*
         *  Gradient should be w.r.t. center_disp
         *  No gradient for center
         */

        if (is_center) {
            return 0.0;
        }

        scalar_t rel_dis = neighbour_disp - center_disp;
        scalar_t scatter_dis = abs(center_disp) * lens_effect;

        scalar_t in_focal = (1.0-myexp(-scatter_dis * scatter_dis * 3.0));
        scalar_t rel_occ  = 0.5 + 0.5 * tanh(10.0 * (rel_dis-0.1));

        scalar_t drel_dis     = -1.0;
        scalar_t dscatter_dis = lens_effect * sign(center_disp);

        scalar_t d_in_focal = -myexp(-3.0 * scatter_dis * scatter_dis) * (-6.0 * scatter_dis) * dscatter_dis;
        scalar_t d_rel_occ  = 0.5 * d_tanh(10.0 * (rel_dis-0.1)) * 10.0 * drel_dis;


        d_in_focal = clip_grad(d_in_focal);
        d_rel_occ  = clip_grad(d_rel_occ);

        return (1.0-in_focal) * d_rel_occ + d_in_focal * (1.0-rel_occ);
    }


    template <typename scalar_t>
    __device__ __forceinline__ scalar_t scatter_weight_no_occlusion(
            scalar_t lens_shape,
            scalar_t lens_effect,
            scalar_t disp,
            scalar_t rel_dis) {

        scalar_t scatter_dis   = abs(disp) * lens_effect + 1.0;
        scalar_t scatter       = softstep(scatter_dis, rel_dis);
        scalar_t area_reweight = 1.0 / (scatter_dis * scatter_dis);

        scalar_t weight = lens_shape * area_reweight * scatter;
        return  weight;
    }

    /*
     * Derivaties of the scattering weight function
     * */
    template <typename scalar_t>
    __device__ __forceinline__ scalar_t d_scatter_no_occlusion_weight(
            scalar_t lens_shape, 
            scalar_t lens_effect,
            scalar_t disp, 
            scalar_t rel_dis) {

        scalar_t scatter_dis =  abs(disp) * lens_effect + 1.0;

        scalar_t area_reweight = 1.0 / (scatter_dis * scatter_dis);
        scalar_t scatter = softstep(scatter_dis, rel_dis); 

        scalar_t dscatter_dis = lens_effect * sign(disp);
        scalar_t dscatter = d_softstep(scatter_dis, rel_dis) * dscatter_dis;
        scalar_t darea_reweight = -2.0 * 1.0/(scatter_dis * scatter_dis * scatter_dis) * dscatter_dis;  

        return lens_shape * (dscatter * area_reweight + scatter * darea_reweight);
    }


    template <typename scalar_t>
    __device__ __forceinline__ scalar_t scatter_weight(
            scalar_t lens_shape,
            scalar_t lens_effect,
            scalar_t disp,
            scalar_t rel_dis,
            scalar_t center_disp,
            bool     is_center) {

        scalar_t scatter_dis   = abs(disp) * lens_effect + 1.0;
        // scalar_t scatter       = softstep(scatter_dis, rel_dis);

        scalar_t scatter = softstep(scatter_dis, rel_dis);
        scalar_t area_reweight = 1.0 / (scatter_dis * scatter_dis);

        scalar_t weight = lens_shape * area_reweight * scatter;

        return  weight;
    }

    /*
     * Derivaties of the scattering weight function
     * */
    template <typename scalar_t>
    __device__ __forceinline__ scalar_t d_scatter_weight(
            scalar_t lens_shape,
            scalar_t lens_effect,
            scalar_t disp,
            scalar_t rel_dis,
            scalar_t center_disp,
            bool     is_center) {

        scalar_t scatter_dis   = abs(disp) * lens_effect + 1.0;
        scalar_t area_reweight = 1.0 / (scatter_dis * scatter_dis);
        // scalar_t scatter       = softstep(scatter_dis, rel_dis);
        scalar_t scatter = softstep(scatter_dis , rel_dis);

        scalar_t dscatter_dis   = lens_effect * sign(disp);
        // scalar_t dscatter       = d_softstep(scatter_dis, rel_dis) * dscatter_dis;
        scalar_t dscatter       = d_softstep(scatter_dis, rel_dis) * dscatter_dis;
        scalar_t darea_reweight = -2.0 * 1.0/(scatter_dis * scatter_dis * scatter_dis) * dscatter_dis;

        return lens_shape * (darea_reweight * scatter + dscatter * area_reweight);
    }


    template <typename scalar_t>
    __global__ void scatter_cuda_forward_kernel(
        const torch::PackedTensorAccessor64<scalar_t,4> d_x,
        const torch::PackedTensorAccessor64<scalar_t,2> d_lens_effect,
        const torch::PackedTensorAccessor64<scalar_t,2> d_kernel,
        const torch::PackedTensorAccessor64<scalar_t,2> d_lens_mask,
        torch::PackedTensorAccessor64<scalar_t,4> d_blur) {

        int bi, wi, hi;
        const int wstride = gridDim.x * blockDim.x;
        const int hstride = gridDim.y * blockDim.y;
        const int bstride = gridDim.z * blockDim.z;

        const int half_blur_ksize = d_kernel.size(1)/2;
        const int padding = half_blur_ksize;
        const int batch_size = d_x.size(0);
        const int h = d_x.size(2)-2*padding;
        const int w = d_x.size(3)-2*padding;

        for (bi = blockIdx.z; bi < batch_size; bi += bstride) {
            for (wi = blockIdx.x * blockDim.x + threadIdx.x; wi < w; wi += wstride) /* 0 ~ w */
                for(hi = blockIdx.y * blockDim.y + threadIdx.y; hi < h; hi += hstride) {
                    int img_hi = hi + padding;
                    int img_wi = wi + padding;

                    scalar_t lens_effect = d_lens_effect[bi][0];
                    scalar_t r(0.0), g(0.0), b(0.0), weight(0.0), o(0.0);
                    scalar_t center_alpha = d_x[bi][3][img_hi][img_wi];
                    scalar_t center_disp  = d_x[bi][4][img_hi][img_wi];

                    /* neighbourhood scattering */
                    for(int lhi = -half_blur_ksize; lhi <= half_blur_ksize; lhi++) {
                        for(int lwi = -half_blur_ksize; lwi <= half_blur_ksize; ++lwi) {
                        /* Padding */
                        int local_hi = img_hi + lhi;
                        int local_wi = img_wi + lwi;
                        int ker_hi   = lhi + half_blur_ksize;
                        int ker_wi   = lwi + half_blur_ksize;

                        scalar_t krel_dis    = d_kernel[ker_hi][ker_wi];
                        scalar_t lens_mask   = d_lens_mask[ker_hi][ker_wi];

                        scalar_t cur_r    = d_x[bi][0][local_hi][local_wi];
                        scalar_t cur_g    = d_x[bi][1][local_hi][local_wi];
                        scalar_t cur_b    = d_x[bi][2][local_hi][local_wi];
                        scalar_t cur_a    = d_x[bi][3][local_hi][local_wi];
                        scalar_t cur_disp = d_x[bi][4][local_hi][local_wi];

                        scalar_t cur_weight = scatter_weight(lens_mask, lens_effect, cur_disp, krel_dis, center_disp, false);
                        scalar_t occlusion  = soft_occlusion(center_disp, cur_disp, lens_effect, false);

                        if (lhi == 0 && lwi == 0) {
                            cur_weight = scatter_weight(lens_mask, lens_effect, cur_disp, krel_dis, center_disp, true);
                            occlusion  = soft_occlusion(center_disp, cur_disp, lens_effect, true);
                        }

                        cur_weight = cur_weight * occlusion * cur_a;

                        r += cur_r * cur_weight;
                        g += cur_g * cur_weight;
                        b += cur_b * cur_weight;

                        weight += cur_weight;
                    }
                }

                scalar_t scatter_dis = abs(center_disp) * lens_effect + 1.0;
                int search_range = min((int)(scatter_dis/2), half_blur_ksize);

                for (int lhi = -search_range; lhi <= search_range; ++lhi) {
                    for (int lwi = -search_range; lwi <= search_range; ++lwi) {
                        int      local_hi = img_hi + lhi;
                        int      local_wi = img_wi + lwi;
                        scalar_t cur_a    = d_x[bi][3][local_hi][local_wi];

                        o += cur_a;
                    }
                }

                o = o / ( (2 * search_range + 1) * (2 * search_range + 1));

                // weighta = 1.0;
                d_blur[bi][0][hi][wi] = r;
                d_blur[bi][1][hi][wi] = g;
                d_blur[bi][2][hi][wi] = b;
                d_blur[bi][3][hi][wi] = weight;
                d_blur[bi][4][hi][wi] = o;
            }
        }
    }


    template <typename scalar_t>
    __global__ void scatter_cuda_backward_kernel(
        const torch::PackedTensorAccessor64<scalar_t,4> d_drgbw,
        const torch::PackedTensorAccessor64<scalar_t,4> d_x,
        const torch::PackedTensorAccessor64<scalar_t,4> d_blur,
        const torch::PackedTensorAccessor64<scalar_t,2> d_lens_effect,
        const torch::PackedTensorAccessor64<scalar_t,2> d_kernel,
        const torch::PackedTensorAccessor64<scalar_t,2> d_lens_mask,
        torch::PackedTensorAccessor64<scalar_t,4> d_dx) {

        const int bstart  = blockIdx.z;
        const int wstart  = blockIdx.x * blockDim.x + threadIdx.x;
        const int hstart  = blockIdx.y * blockDim.y + threadIdx.y;
        const int wstride = gridDim.x * blockDim.x;
        const int hstride = gridDim.y * blockDim.y;
        const int bstride = gridDim.z * blockDim.z;

        const int half_blur_ksize = d_kernel.size(1)/2;
        const int padding         = half_blur_ksize;
        const int batch_size      = d_x.size(0);
        const int h               = d_x.size(2)-2*padding;
        const int w               = d_x.size(3)-2*padding;

        for (int bi = bstart; bi < batch_size; bi += bstride) {
            for (int wi = wstart; wi < w; wi += wstride) /* 0 ~ w */
                for(int hi = hstart; hi < h; hi += hstride) {
                /* Note, refer to scattering equation for details */
                /* d_I = \sum I(x) * (d_w(x+dx) * (1.0/W_i(x+dx) - 1.0/W_i(x+dx)^2)) */
                size_t img_hi = padding + hi;
                size_t img_wi = padding + wi;

                scalar_t r        = d_x[bi][0][img_hi][img_wi];
                scalar_t g        = d_x[bi][1][img_hi][img_wi];
                scalar_t b        = d_x[bi][2][img_hi][img_wi];
                scalar_t cur_a    = d_x[bi][3][img_hi][img_wi];
                scalar_t cur_disp = d_x[bi][4][img_hi][img_wi];

                scalar_t dy_r_c = d_drgbw[bi][0][img_hi][img_wi];
                scalar_t dy_g_c = d_drgbw[bi][1][img_hi][img_wi];
                scalar_t dy_b_c = d_drgbw[bi][2][img_hi][img_wi];
                scalar_t dy_W_c = d_drgbw[bi][3][img_hi][img_wi];

                scalar_t dr = 0.0;
                scalar_t dg = 0.0;
                scalar_t db = 0.0;
                scalar_t da = 0.0;
                scalar_t dd = 0.0;

                scalar_t lens_effect = d_lens_effect[bi][0];

                for(int lhi = -half_blur_ksize; lhi <= half_blur_ksize; lhi++)
                    for(int lwi = -half_blur_ksize; lwi <= half_blur_ksize; lwi++) {
                    bool is_center = false;

                    if (lhi == 0 && lwi == 0) {
                        is_center = true;
                    }

                    /* Indices in kernel space */
                    size_t ker_hi = lhi + half_blur_ksize;
                    size_t ker_wi = lwi + half_blur_ksize;

                    /* Indices in image space */
                    size_t local_hi = img_hi + lhi;
                    size_t local_wi = img_wi + lwi;

                    scalar_t lens_mask = d_lens_mask[ker_hi][ker_wi];
                    scalar_t krel_dis  = d_kernel[ker_hi][ker_wi];

                    scalar_t local_r = d_x[bi][0][local_hi][local_wi];
                    scalar_t local_g = d_x[bi][1][local_hi][local_wi];
                    scalar_t local_b = d_x[bi][2][local_hi][local_wi];
                    scalar_t local_a = d_x[bi][3][local_hi][local_wi];
                    scalar_t local_d = d_x[bi][4][local_hi][local_wi];

                    scalar_t dy_r  = d_drgbw[bi][0][local_hi][local_wi];
                    scalar_t dy_g  = d_drgbw[bi][1][local_hi][local_wi];
                    scalar_t dy_b  = d_drgbw[bi][2][local_hi][local_wi];
                    scalar_t dy_W  = d_drgbw[bi][3][local_hi][local_wi];
                    scalar_t dy_O  = d_drgbw[bi][4][local_hi][local_wi];

                    scalar_t cur_weight = scatter_weight(lens_mask, lens_effect, cur_disp, krel_dis, local_d, is_center);
                    scalar_t occlusion  = soft_occlusion(local_d, cur_disp, lens_effect, is_center);

                    scalar_t dw_dd   = d_scatter_weight(lens_mask, lens_effect, cur_disp, krel_dis, local_d, is_center);
                    scalar_t do_dd   = d_soft_occlusion_nd(local_d, cur_disp, lens_effect, is_center);
                    scalar_t do_dd_c = d_soft_occlusion_cd(cur_disp, local_d, lens_effect, is_center);

                    /* Alpha backward */
                    scalar_t scatter_dis = abs(local_d) * lens_effect + 1.0;
                    int search_range = min((int)(scatter_dis/2), padding);
                    scalar_t norm_term = 1.0/((2 * search_range + 1) * (2 * search_range + 1));

                    if (search_range >= abs(lhi) && search_range >= abs(lwi)) { /* Inside the searching range */
                        da += dy_O * norm_term;
                    }

                    scalar_t dr_dw = r * cur_a;
                    scalar_t dg_dw = g * cur_a;
                    scalar_t db_dw = b * cur_a;

                    scalar_t dW_da = cur_weight * occlusion;

                    dr += (dy_r) * cur_weight * occlusion * cur_a;
                    dg += (dy_g) * cur_weight * occlusion * cur_a;
                    db += (dy_b) * cur_weight * occlusion * cur_a;

                    da += (dy_r * r  + dy_g * g  + dy_b * b) * cur_weight * occlusion;
                    da += dy_W * dW_da;

                    dd += (dy_r * dr_dw + dy_g * dg_dw + dy_b * db_dw) * (dw_dd * occlusion + cur_weight * do_dd);
                    dd += dy_W * cur_a * (dw_dd * occlusion + cur_weight * do_dd);

                    /* w.r.t. center disparity */
                    cur_weight = scatter_weight(lens_mask, lens_effect, local_d, krel_dis, cur_disp, is_center);
                    dd += (dy_r_c * local_r + dy_g_c * local_g + dy_b_c * local_b) * local_a * cur_weight * do_dd_c;
                    dd += dy_W_c * local_a * cur_weight * do_dd_c;
                }


                /* Gradient w.r.t. RGBAD
                */
                d_dx[bi][0][hi][wi] = dr;
                d_dx[bi][1][hi][wi] = dg;
                d_dx[bi][2][hi][wi] = db;
                d_dx[bi][3][hi][wi] = da;
                d_dx[bi][4][hi][wi] = dd;
            }
        }
    }


    template <typename scalar_t>
    __global__ void scatter_cuda_no_occlusion_forward_kernel(
        const torch::PackedTensorAccessor64<scalar_t,4> d_x,
        const torch::PackedTensorAccessor64<scalar_t,2> d_lens_effect,
        const torch::PackedTensorAccessor64<scalar_t,2> d_kernel,
        const torch::PackedTensorAccessor64<scalar_t,2> d_lens_mask,
        torch::PackedTensorAccessor64<scalar_t,4> d_blur,
        torch::PackedTensorAccessor64<scalar_t,4> d_weights) {

        int bi, wi, hi;
        const int wstride = gridDim.x * blockDim.x;
        const int hstride = gridDim.y * blockDim.y;
        const int bstride = gridDim.z * blockDim.z;

        const int half_blur_ksize = d_kernel.size(1)/2;
        const int padding = half_blur_ksize;
        const int batch_size = d_x.size(0);
        const int h = d_x.size(2)-2*padding;
        const int w = d_x.size(3)-2*padding;

        for (bi = blockIdx.z; bi < batch_size; bi += bstride) {
            for (wi = blockIdx.x * blockDim.x + threadIdx.x; wi < w; wi += wstride)
                for(hi = blockIdx.y * blockDim.y + threadIdx.y; hi < h; hi += hstride) {

                scalar_t r(0.0), g(0.0), b(0.0), weight(0.0);
                scalar_t center_disp = d_x[bi][3][hi][wi];

                /* neighbourhood scattering */
                for(int lhi = -half_blur_ksize; lhi <= half_blur_ksize; lhi++)
                    for(int lwi = -half_blur_ksize; lwi <= half_blur_ksize; ++lwi) {
                    /* Padding */
                    int curhi = hi + lhi;
                    int curwi = wi + lwi;

                    scalar_t cur_disp    = d_x[bi][3][padding + curhi][padding + curwi];
                    scalar_t krel_dis    = d_kernel[lhi+half_blur_ksize][lwi+half_blur_ksize];
                    scalar_t lens_mask   = d_lens_mask[lhi+half_blur_ksize][lwi+half_blur_ksize];
                    scalar_t lens_effect = d_lens_effect[bi][0];

                    scalar_t cur_weight = scatter_weight_no_occlusion(lens_mask, lens_effect, cur_disp, krel_dis);

                    r += d_x[bi][0][padding + curhi][padding + curwi] * cur_weight;
                    g += d_x[bi][1][padding + curhi][padding + curwi] * cur_weight;
                    b += d_x[bi][2][padding + curhi][padding + curwi] * cur_weight;

                    weight += cur_weight;
                }

                // weight = 1.0;
                d_blur[bi][0][hi][wi]       = r/weight;
                d_blur[bi][1][hi][wi]       = g/weight;
                d_blur[bi][2][hi][wi]       = b/weight;
                d_weights[bi][0][hi][wi]    = weight;
            }
        }
    }

    /* d_dx: gradient w.r.t. original depth input, B x 1 x H x W
    */
    template <typename scalar_t>
    __global__ void scatter_cuda_no_occlusion_backward_kernel(
        const torch::PackedTensorAccessor64<scalar_t,4> d_drgb,
        const torch::PackedTensorAccessor64<scalar_t,4> d_x,
        const torch::PackedTensorAccessor64<scalar_t,4> d_blur,
        const torch::PackedTensorAccessor64<scalar_t,4> d_weights,
        const torch::PackedTensorAccessor64<scalar_t,2> d_lens_effect,
        const torch::PackedTensorAccessor64<scalar_t,2> d_kernel,
        const torch::PackedTensorAccessor64<scalar_t,2> d_lens_mask,
        torch::PackedTensorAccessor64<scalar_t,4> d_dx) {

        const int bstart = blockIdx.z;
        const int wstart = blockIdx.x * blockDim.x + threadIdx.x;
        const int hstart = blockIdx.y * blockDim.y + threadIdx.y;
        const int wstride = gridDim.x * blockDim.x;
        const int hstride = gridDim.y * blockDim.y;
        const int bstride = gridDim.z * blockDim.z;

        const int half_blur_ksize = d_kernel.size(1)/2;
        const int padding = half_blur_ksize;
        const int batch_size = d_x.size(0);
        const int h = d_x.size(2)-2*padding;
        const int w = d_x.size(3)-2*padding;

        for (int bi = bstart; bi < batch_size; bi += bstride) {
            for (int wi = wstart; wi < w; wi += wstride)
                for(int hi = hstart; hi < h; hi += hstride) {
                /* Note, refer to scattering equation for details */
                /* d_I = \sum I(x) * (d_w(x+dx) * (1.0/W_i(x+dx) - 1.0/W_i(x+dx)^2)) */
                size_t img_hi = padding + hi;
                size_t img_wi = padding + wi;

                scalar_t cur_disp = d_x[bi][3][img_hi][img_wi];

                // scalar_t rel_dis = abs(cur_disp) * d_lens_effect + 1.0f;
                scalar_t lens_effect = d_lens_effect[bi][0];
                scalar_t rel_dis = cur_disp + 1.0f;
                scalar_t dsign = sign(lens_effect * cur_disp);
                scalar_t area = rel_dis * rel_dis;

                scalar_t r           = d_x[bi][0][img_hi][img_wi];
                scalar_t g           = d_x[bi][1][img_hi][img_wi];
                scalar_t b           = d_x[bi][2][img_hi][img_wi];
                scalar_t center_disp = d_x[bi][3][img_hi][img_wi];

                scalar_t dr = 0.0;
                scalar_t dg = 0.0;
                scalar_t db = 0.0;
                scalar_t dd = 0.0;

                /* A = rel_dis * rel_dis */
                scalar_t dA = 2.0 * rel_dis * dsign * lens_effect;

                scalar_t dydr = d_drgb[bi][0][hi][wi];
                scalar_t dydg = d_drgb[bi][1][hi][wi];
                scalar_t dydb = d_drgb[bi][2][hi][wi];

                for(int lhi = -half_blur_ksize; lhi <= half_blur_ksize; lhi++)
                    for(int lwi = -half_blur_ksize; lwi <= half_blur_ksize; ++lwi) {
                    /* Indices in kernel space */
                    size_t ker_hi = lhi + half_blur_ksize;
                    size_t ker_wi = lwi + half_blur_ksize;

                    /* Indices in image space */
                    size_t local_hi = img_hi + lhi;
                    size_t local_wi = img_wi + lwi;

                    scalar_t local_r = d_blur[bi][0][local_hi][local_wi];
                    scalar_t local_g = d_blur[bi][1][local_hi][local_wi];
                    scalar_t local_b = d_blur[bi][2][local_hi][local_wi];

                    scalar_t dy_r = d_drgb[bi][0][local_hi][local_wi];
                    scalar_t dy_g = d_drgb[bi][1][local_hi][local_wi];
                    scalar_t dy_b = d_drgb[bi][2][local_hi][local_wi];

                    scalar_t cur_W = d_weights[bi][0][local_hi][local_wi];

                    scalar_t lens_mask   = d_lens_mask[ker_hi][ker_wi];
                    scalar_t krel_dis    = d_kernel[ker_hi][ker_wi];

                    scalar_t cur_weight = scatter_weight_no_occlusion(lens_mask, lens_effect, cur_disp, krel_dis);

                    scalar_t dr_dw = (r-local_r)/ cur_W;
                    scalar_t dg_dw = (g-local_g)/ cur_W;
                    scalar_t db_dw = (b-local_b)/ cur_W;

                    scalar_t dw_dd = d_scatter_no_occlusion_weight(lens_mask, lens_effect, cur_disp, krel_dis);

                    dr += dy_r * cur_weight / cur_W;
                    dg += dy_g * cur_weight / cur_W;
                    db += dy_b * cur_weight / cur_W;

                    /*
                       r = (r0 * w0 + r1 * w1 + ...)/(w0 + w1 + ...)
                       dr/dd0 = r0/W * dw0/dd0 - (r0 * w0 + r1 * w1 + ...)/(W * W) * dw0/dd0
                     */
                    dd += (dy_r * dr_dw + dy_g * dg_dw + dy_b * db_dw) * dw_dd;
                }

                /* Gradient w.r.t. RGBD
                */
                d_dx[bi][0][hi][wi] = dr;
                d_dx[bi][1][hi][wi] = dg;
                d_dx[bi][2][hi][wi] = db;
                d_dx[bi][3][hi][wi] = dd;
            }
        }
    }

} // namespace


torch::Tensor scatter_cuda_forward(torch::Tensor x,
                                   torch::Tensor lens_effect,
                                   torch::Tensor kernel,
                                   torch::Tensor lens_mask) {
    const auto batch_size = x.size(0);
    const auto channel_size = x.size(1);
    const auto h = x.size(2);
    const auto w = x.size(3);
    const auto ksize = kernel.size(0);

    const int threads = 512;
    const dim3 blocks((w + threads - 1) / threads, (h+threads-1)/threads, batch_size);

    torch::Tensor blur_tensor = torch::zeros({batch_size, 5, h-ksize+1, w-ksize+1}).to(x);
    // torch::Tensor blur_tensor = torch::zeros({batch_size, 4, h-ksize+1, w-ksize+1}).to(x);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "scatter_forward_cuda", ([&] {
        scatter_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            x.packed_accessor64<scalar_t,4>(),
            lens_effect.packed_accessor64<scalar_t,2>(),
            kernel.packed_accessor64<scalar_t,2>(),
            lens_mask.packed_accessor64<scalar_t,2>(),
            blur_tensor.packed_accessor64<scalar_t,4>());
    }));

    return blur_tensor;
}


torch::Tensor scatter_cuda_backward(torch::Tensor drgbw,
                                    torch::Tensor x,
                                    torch::Tensor blur,
                                    torch::Tensor lens_effect,
                                    torch::Tensor kernel,
                                    torch::Tensor lens_mask) {
    const auto batch_size = x.size(0);
    const auto channel_size = x.size(1);
    const auto h = x.size(2);
    const auto w = x.size(3);
    const auto ksize = kernel.size(0);
    const int threads = 512;
    const dim3 blocks((w + threads - 1) / threads, (h+threads-1)/threads, batch_size);

    /*
     * dx: RGBA-D
     */
    torch::Tensor dx = torch::zeros({batch_size, 5, h-ksize+1, w-ksize+1}).to(x);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "scatter_forward_cuda", ([&] {
        scatter_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
            drgbw.packed_accessor64<scalar_t,4>(),
            x.packed_accessor64<scalar_t,4>(),
            blur.packed_accessor64<scalar_t,4>(),
            lens_effect.packed_accessor64<scalar_t,2>(),
            kernel.packed_accessor64<scalar_t,2>(),
            lens_mask.packed_accessor64<scalar_t,2>(),
            dx.packed_accessor64<scalar_t,4>());
    }));

    return dx;
}


std::vector<torch::Tensor> scatter_cuda_no_occlusion_forward(torch::Tensor x,
                                                             torch::Tensor lens_effect,
                                                             torch::Tensor kernel,
                                                             torch::Tensor lens_mask) {
    const auto batch_size = x.size(0);
    const auto channel_size = x.size(1);
    const auto h = x.size(2);
    const auto w = x.size(3);
    const auto ksize = kernel.size(0);
    const int threads = 512;
    const dim3 blocks((w + threads - 1) / threads, (h+threads-1)/threads, batch_size);

    torch::Tensor blur_tensor = torch::zeros({batch_size, 3, h-ksize+1, w-ksize+1}).to(x);
    /* weights needs to be padded */
    torch::Tensor weights = torch::zeros({batch_size, 1, h-ksize+1, w-ksize+1}).to(x);
    AT_DISPATCH_FLOATING_TYPES(x.type(), "scatter_forward_cuda", ([&] {
        scatter_cuda_no_occlusion_forward_kernel<scalar_t><<<blocks, threads>>>(
            x.packed_accessor64<scalar_t,4>(),
            lens_effect.packed_accessor64<scalar_t,2>(),
            kernel.packed_accessor64<scalar_t,2>(),
            lens_mask.packed_accessor64<scalar_t,2>(),
            blur_tensor.packed_accessor64<scalar_t,4>(),
            weights.packed_accessor64<scalar_t,4>());
    }));

    return {blur_tensor, weights};
}


torch::Tensor scatter_cuda_no_occlusion_backward(torch::Tensor drgb,
                                                 torch::Tensor x,
                                                 torch::Tensor blur,
                                                 torch::Tensor weights,
                                                 torch::Tensor lens_effect,
                                                 torch::Tensor kernel,
                                                 torch::Tensor lens_mask) {
    const auto batch_size = x.size(0);
    const auto channel_size = x.size(1);
    const auto h = x.size(2);
    const auto w = x.size(3);
    const auto ksize = kernel.size(0);
    const int threads = 512;
    const dim3 blocks((w + threads - 1) / threads, (h+threads-1)/threads, batch_size);

    torch::Tensor dx = torch::zeros({batch_size, 4, h-ksize+1, w-ksize+1}).to(x);
    AT_DISPATCH_FLOATING_TYPES(x.type(), "scatter_forward_cuda", ([&] {
        scatter_cuda_no_occlusion_backward_kernel<scalar_t><<<blocks, threads>>>(
            drgb.packed_accessor64<scalar_t,4>(),
            x.packed_accessor64<scalar_t,4>(),
            blur.packed_accessor64<scalar_t,4>(),
            weights.packed_accessor64<scalar_t,4>(),
            lens_effect.packed_accessor64<scalar_t,2>(),
            kernel.packed_accessor64<scalar_t,2>(),
            lens_mask.packed_accessor64<scalar_t,2>(),
            dx.packed_accessor64<scalar_t,4>());
    }));

    return dx;
}
