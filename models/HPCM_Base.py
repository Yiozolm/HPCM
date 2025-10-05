import torch
import torch.nn as nn

from compressai.ops import quantize_ste

from .Base import BaseWrapper

from .utils import ( 
    conv4x4_down, 
    PConvRB, 
    conv2x2_down, 
    deconv2x2_up, 
    deconv4x4_up, 
    conv1x1,
    y_spatial_prior_s1_s2, 
    y_spatial_prior_s3,
    CrossAttentionCell, 
)


class HPCM_Base(BaseWrapper):
    def __init__(self, N=256, M=320, **kwargs):
        super().__init__()

        mlp_ratio = kwargs.get("mlp_ratio", 4)
        partial_ratio = kwargs.get("partial_ratio", 4)

        self.g_a = nn.Sequential(
            conv4x4_down(3,96),
            PConvRB(96, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(96, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            conv2x2_down(96,192),
            PConvRB(192, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(192, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            conv2x2_down(192,384),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            conv2x2_down(384,M),
        )

        self.g_s = nn.Sequential(
            deconv2x2_up(M,384),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            deconv2x2_up(384,192),
            PConvRB(192, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(192, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            deconv2x2_up(192,96),
            PConvRB(96, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(96, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            deconv4x4_up(96,3),
        )

        self.h_a = nn.Sequential(
            PConvRB(M, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            conv2x2_down(M,N),
            PConvRB(N, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(N, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(N, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            conv2x2_down(N,N),
        )

        self.h_s = nn.Sequential(
            deconv2x2_up(N,N),
            PConvRB(N, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(N, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(N, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            deconv2x2_up(N,M*2),
            PConvRB(M*2, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
        )

        self.means_hyper = nn.Parameter(torch.zeros(1,N,1,1))
        self.scales_hyper = nn.Parameter(torch.ones(1,N,1,1))

        self.y_spatial_prior_adaptor_list_s1 = nn.ModuleList(conv1x1(3*M,3*M) for _ in range(1))
        self.y_spatial_prior_adaptor_list_s2 = nn.ModuleList(conv1x1(3*M,3*M) for _ in range(3))
        self.y_spatial_prior_adaptor_list_s3 = nn.ModuleList(conv1x1(3*M,3*M) for _ in range(6))
        self.y_spatial_prior_s1_s2 = y_spatial_prior_s1_s2(M)
        self.y_spatial_prior_s3 = y_spatial_prior_s3(M)

        self.adaptive_params_list = [
            torch.nn.Parameter(torch.ones((1, M*3, 1, 1), device='cuda'), requires_grad=True) for _ in range(10)
        ]

        self.attn_s1 = CrossAttentionCell(320*2, 320*2, window_size=4, kernel_size=1)
        self.attn_s2 = CrossAttentionCell(320*2, 320*2, window_size=8, kernel_size=1)
        self.attn_s3 = CrossAttentionCell(320*2, 320*2, window_size=8, kernel_size=1)
        
        self.context_net = nn.ModuleList(conv1x1(2*M,2*M) for _ in range(2))

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        _, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = quantize_ste(z - self.means_hyper) + self.means_hyper

        params = self.h_s(z_hat)
        y_res, y_q, y_hat, scales_y = self._forward_hpcm(y, params, 
                                self.y_spatial_prior_adaptor_list_s1, self.y_spatial_prior_s1_s2, 
                                self.y_spatial_prior_adaptor_list_s2, self.y_spatial_prior_s1_s2, 
                                self.y_spatial_prior_adaptor_list_s3, self.y_spatial_prior_s3, 
                                self.adaptive_params_list, self.context_net, 
                                )
        _, y_likelihoods = self.gaussian_conditional(y_res, scales_y)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def _forward_hpcm(self, y, common_params, 
                              y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                              y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                              y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                              adaptive_params_list, context_net, write=False):
        B, C, H, W = y.size()
        dtype = common_params.dtype
        device = common_params.device

        ############### 2-step scale-1 (s1) (4× downsample) coding
        # get y_s2 first
        mask_list_s2 = self.get_mask_for_s2(B, C, H, W, dtype, device)
        y_s2 = self.get_s1_s2_with_mask(y, mask_list_s2, B, C, H // 2, W // 2, reduce=8)
        # get y_s1 from y_s2
        mask_list_rec_s2 = self.get_mask_for_rec_s2(B, C, H // 2, W // 2, dtype, device)
        y_s1 = self.get_s1_s2_with_mask(y_s2, mask_list_rec_s2, B, C, H // 4, W // 4, reduce=4)

        # get scales_s1 and means_s1, same as getting s1 and s2
        scales_all, means_all = common_params.chunk(2,1)
        scales_s2 = self.get_s1_s2_with_mask(scales_all, mask_list_s2, B, C, H // 2, W // 2, reduce=8)
        scales_s1 = self.get_s1_s2_with_mask(scales_s2, mask_list_rec_s2, B, C, H // 4, W // 4, reduce=4)
        means_s2 = self.get_s1_s2_with_mask(means_all, mask_list_s2, B, C, H // 2, W // 2, reduce=8)
        means_s1 = self.get_s1_s2_with_mask(means_s2, mask_list_rec_s2, B, C, H // 4, W // 4, reduce=4)
        common_params_s1 = torch.cat((scales_s1, means_s1), dim=1)
        context_next = common_params_s1

        mask_list = self.get_mask_two_parts(B, C, H // 4, W // 4, dtype, device)
        y_res_list_s1 = []
        y_q_list_s1 = []
        y_hat_list_s1 = []
        scale_list_s1 = []

        for i in range(2):
            if i == 0:
                y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y_s1, scales_s1, means_s1, mask_list[i])
                y_res_list_s1.append(y_res_0)
                y_q_list_s1.append(y_q_0)
                y_hat_list_s1.append(y_hat_0)
                scale_list_s1.append(s_hat_0)
            else:
                y_hat_so_far = torch.sum(torch.stack(y_hat_list_s1), dim=0)
                params = torch.cat((context_next, y_hat_so_far), dim=1)
                context = self.y_spatial_prior_s1_s2(self.y_spatial_prior_adaptor_list_s1[i - 1](params), self.adaptive_params_list[i - 1])
                context_next = self.attn_s1(context, context_next)
                scales, means = context.chunk(2, 1)
                y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y_s1, scales, means, mask_list[i])
                y_res_list_s1.append(y_res_1)
                y_q_list_s1.append(y_q_1)
                y_hat_list_s1.append(y_hat_1)
                scale_list_s1.append(s_hat_1)
        
        y_res = torch.sum(torch.stack(y_res_list_s1), dim=0)
        y_q = torch.sum(torch.stack(y_q_list_s1), dim=0)
        y_hat = torch.sum(torch.stack(y_hat_list_s1), dim=0)
        scales_hat = torch.sum(torch.stack(scale_list_s1), dim=0)

        if write:
            y_q_write_list_s1 = [self.combine_for_writing_s1(y_q_list_s1[i]) for i in range(len(y_q_list_s1))]
            scales_hat_write_list_s1 = [self.combine_for_writing_s1(scale_list_s1[i]) for i in range(len(scale_list_s1))]
        
        # up-scaling to s2
        y_res = self.recon_for_s2_s3(y_res, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        y_q = self.recon_for_s2_s3(y_q, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        y_hat = self.recon_for_s2_s3(y_hat, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        scales_hat = self.recon_for_s2_s3(scales_hat, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.recon_for_s2_s3(context_next_scales, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        context_next_means = self.recon_for_s2_s3(context_next_means, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 4-step scale-2 (s2) (2× downsample) coding

        mask_list_s1 = self.get_mask_for_s1(B, C, H, W, dtype, device)
        scales_s2 = self.get_s2_hyper_with_mask(scales_all, mask_list_s1, mask_list_s2, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        means_s2 = self.get_s2_hyper_with_mask(means_all, mask_list_s1, mask_list_s2, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        common_params_s2 = torch.cat((scales_s2, means_s2), dim=1)
        context += common_params_s2
        context_next = context_net[0](context)
        
        mask_list = self.get_mask_four_parts(B, C, H // 2, W // 2, dtype, device)[1:]
        y_res_list_s2 = [y_res]
        y_q_list_s2   = [y_q]
        y_hat_list_s2 = [y_hat]
        scale_list_s2 = [scales_hat]

        for i in range(3):
            y_hat_so_far = torch.sum(torch.stack(y_hat_list_s2), dim=0)
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context = self.y_spatial_prior_s1_s2(self.y_spatial_prior_adaptor_list_s2[i - 1](params), self.adaptive_params_list[i + 1])
            context_next = self.attn_s2(context, context_next)
            scales, means = context.chunk(2, 1)
            y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y_s2, scales, means, mask_list[i])
            y_res_list_s2.append(y_res_1)
            y_q_list_s2.append(y_q_1)
            y_hat_list_s2.append(y_hat_1)
            scale_list_s2.append(s_hat_1)
        
        y_res = torch.sum(torch.stack(y_res_list_s2), dim=0)
        y_q = torch.sum(torch.stack(y_q_list_s2), dim=0)
        y_hat = torch.sum(torch.stack(y_hat_list_s2), dim=0)
        scales_hat = torch.sum(torch.stack(scale_list_s2), dim=0)

        if write:
            y_q_write_list_s2 = [self.combine_for_writing_s2(y_q_list_s2[i]) for i in range(1, len(y_q_list_s2))]
            scales_hat_write_list_s2 = [self.combine_for_writing_s2(scale_list_s2[i]) for i in range(1, len(scale_list_s2))]
       
        # up-scaling to s3
        y_res = self.recon_for_s2_s3(y_res, mask_list_s2, B, C, H, W, dtype, device)
        y_q = self.recon_for_s2_s3(y_q, mask_list_s2, B, C, H, W, dtype, device)
        y_hat = self.recon_for_s2_s3(y_hat, mask_list_s2, B, C, H, W, dtype, device)
        scales_hat = self.recon_for_s2_s3(scales_hat, mask_list_s2, B, C, H, W, dtype, device)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.recon_for_s2_s3(context_next_scales, mask_list_s2, B, C, H, W, dtype, device)
        context_next_means = self.recon_for_s2_s3(context_next_means, mask_list_s2, B, C, H, W, dtype, device)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 8-step scale-3 (s3) coding

        scales_s3 = self.get_s3_hyper_with_mask(scales_all, mask_list_s2, B, C, H, W, dtype, device)
        means_s3 = self.get_s3_hyper_with_mask(means_all, mask_list_s2, B, C, H, W, dtype, device)
        common_params_s3 = torch.cat((scales_s3, means_s3), dim=1)
        context += common_params_s3
        context_next = context_net[1](context)

        mask_list = self.get_mask_eight_parts(B, C, H, W, dtype, device)[2:]
        y_res_list_s3 = [y_res]
        y_q_list_s3   = [y_q]
        y_hat_list_s3 = [y_hat]
        scale_list_s3 = [scales_hat]

        for i in range(6):
            y_hat_so_far = torch.sum(torch.stack(y_hat_list_s3), dim=0)
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context = self.y_spatial_prior_s3(self.y_spatial_prior_adaptor_list_s3[i - 1](params), self.adaptive_params_list[i + 4])
            context_next = self.attn_s3(context, context_next)
            scales, means = context.chunk(2, 1)
            y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_list[i])
            y_res_list_s3.append(y_res_1)
            y_q_list_s3.append(y_q_1)
            y_hat_list_s3.append(y_hat_1)
            scale_list_s3.append(s_hat_1)

        y_res = torch.sum(torch.stack(y_res_list_s3), dim=0)
        y_q = torch.sum(torch.stack(y_q_list_s3), dim=0)
        y_hat = torch.sum(torch.stack(y_hat_list_s3), dim=0)
        scales_hat = torch.sum(torch.stack(scale_list_s3), dim=0)

        if write:
            y_q_write_list_s3 = [self.combine_for_writing_s3(y_q_list_s3[i]) for i in range(1, len(y_q_list_s3))]
            scales_hat_write_list_s3 = [self.combine_for_writing_s3(scale_list_s3[i]) for i in range(1, len(scale_list_s3))]

            return y_q_write_list_s1 + y_q_write_list_s2 + y_q_write_list_s3, scales_hat_write_list_s1 + scales_hat_write_list_s2 + scales_hat_write_list_s3

        return y_res, y_q, y_hat, scales_hat
    
    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z - self.means_hyper)
        z_hat = quantize_ste(z - self.means_hyper) + self.means_hyper

        params = self.h_s(z_hat)

        y_q_write_list, scales_hat_write_list = self._forward_hpcm(y, params, 
                                        self.y_spatial_prior_adaptor_list_s1, self.y_spatial_prior_s1_s2, 
                                        self.y_spatial_prior_adaptor_list_s2, self.y_spatial_prior_s1_s2, 
                                        self.y_spatial_prior_adaptor_list_s3, self.y_spatial_prior_s3, 
                                        self.adaptive_params_list, self.context_net, write=True
                                        )
        
        y_strings = list()
        for i in range(len(y_q_write_list)):
            indexes = self.gaussian_conditional.build_indexes(scales=scales_hat_write_list[i])
            y_string = self.gaussian_conditional.compress(y_q_write_list[i], indexes)
            y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    
    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape) + self.means_hyper

        params = self.h_s(z_hat)
        _, means = params.chunk(2,1)
        B, C, H, W = means.size()
        dtype = means.dtype
        device = means.device

        ############### 2-step resolution-1 (s1) (4× downsample) coding
        mask_list_s2 = self.get_mask_for_s2(B, C, H, W, dtype, device)
        mask_list_rec_s2 = self.get_mask_for_rec_s2(B, C, H // 2, W // 2, dtype, device)

        # get scales_s1 and means_s1
        scales_all, means_all = params.chunk(2,1)
        scales_s2 = self.get_s1_s2_with_mask(scales_all, mask_list_s2, B, C, H // 2, W // 2, reduce=8)
        scales_s1 = self.get_s1_s2_with_mask(scales_s2, mask_list_rec_s2, B, C, H // 4, W // 4, reduce=4)
        means_s2 = self.get_s1_s2_with_mask(means_all, mask_list_s2, B, C, H // 2, W // 2, reduce=8)
        means_s1 = self.get_s1_s2_with_mask(means_s2, mask_list_rec_s2, B, C, H // 4, W // 4, reduce=4)
        common_params_s1 = torch.cat((scales_s1, means_s1), dim=1)
        context_next = common_params_s1

        mask_list = self.get_mask_two_parts(B, C, H // 4, W // 4, dtype, device)

        string_idx = 0
        for i in range(2):
            if i == 0:
                scales_r = self.combine_for_writing_s1(scales_s1 * mask_list[i])
                indexes_r = self.gaussian_conditional.build_indexes(scales=scales_r)
                y_q_r = self.gaussian_conditional.decompress(strings[0][string_idx], indexes_r, dtype).to(device)
                y_hat_curr_step = (torch.cat([y_q_r for _ in range(2)], dim=1) + means_s1) * mask_list[i]
                y_hat_so_far = y_hat_curr_step
                string_idx += 1
            else:
                params = torch.cat((context_next, y_hat_so_far), dim=1)
                context = self.y_spatial_prior_s1_s2(self.y_spatial_prior_adaptor_list_s1[i - 1](params), self.adaptive_params_list[i - 1])
                context_next = self.attn_s1(context, context_next)
                scales, means = context.chunk(2, 1)
                scales_r = self.combine_for_writing_s1(scales * mask_list[i])
                indexes_r = self.gaussian_conditional.build_indexes(scales=scales_r)
                y_q_r = self.gaussian_conditional.decompress(strings[0][string_idx], indexes_r, dtype).to(device)
                y_hat_curr_step = (torch.cat([y_q_r for _ in range(2)], dim=1) + means) * mask_list[i]
                y_hat_so_far = y_hat_so_far + y_hat_curr_step
                string_idx += 1
        
        y_hat_so_far = self.recon_for_s2_s3(y_hat_so_far, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.recon_for_s2_s3(context_next_scales, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        context_next_means = self.recon_for_s2_s3(context_next_means, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 4-step resolution-2 (s2) (2× downsample) coding
        mask_list_s1 = self.get_mask_for_s1(B, C, H, W, dtype, device)
        scales_s2 = self.get_s2_hyper_with_mask(scales_all, mask_list_s1, mask_list_s2, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        means_s2 = self.get_s2_hyper_with_mask(means_all, mask_list_s1, mask_list_s2, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        common_params_s2 = torch.cat((scales_s2, means_s2), dim=1)
        context += common_params_s2
        context_next = self.context_net[0](context)

        mask_list = self.get_mask_four_parts(B, C, H // 2, W // 2, dtype, device)[1:]

        for i in range(3):
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context = self.y_spatial_prior_s1_s2(self.y_spatial_prior_adaptor_list_s2[i - 1](params), self.adaptive_params_list[i + 1])
            context_next = self.attn_s2(context, context_next)
            scales, means = context.chunk(2, 1)
            scales_r = self.combine_for_writing_s2(scales * mask_list[i])
            indexes_r = self.gaussian_conditional.build_indexes(scales=scales_r)
            y_q_r = self.gaussian_conditional.decompress(strings[0][string_idx], indexes_r, dtype).to(device)
            y_hat_curr_step = (torch.cat([y_q_r for _ in range(4)], dim=1) + means) * mask_list[i]
            y_hat_so_far = y_hat_so_far + y_hat_curr_step
            string_idx += 1

        y_hat_so_far = self.recon_for_s2_s3(y_hat_so_far, mask_list_s2, B, C, H, W, dtype, device)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.recon_for_s2_s3(context_next_scales, mask_list_s2, B, C, H, W, dtype, device)
        context_next_means = self.recon_for_s2_s3(context_next_means, mask_list_s2, B, C, H, W, dtype, device)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 8-step resolution-3 (s3) coding
        scales_s3 = self.get_s3_hyper_with_mask(scales_all, mask_list_s2, B, C, H, W, dtype, device)
        means_s3 = self.get_s3_hyper_with_mask(means_all, mask_list_s2, B, C, H, W, dtype, device)
        common_params_s3 = torch.cat((scales_s3, means_s3), dim=1)
        context += common_params_s3
        context_next = self.context_net[1](context)

        mask_list = self.get_mask_eight_parts(B, C, H, W, dtype, device)[2:]

        for i in range(6):
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context = self.y_spatial_prior_s3(self.y_spatial_prior_adaptor_list_s3[i - 1](params), self.adaptive_params_list[i + 4])
            context_next = self.attn_s3(context, context_next)
            scales, means = context.chunk(2, 1)
            scales_r = self.combine_for_writing_s3(scales * mask_list[i])
            indexes_r = self.gaussian_conditional.build_indexes(scales=scales_r)
            y_q_r = self.gaussian_conditional.decompress(strings[0][string_idx], indexes_r, dtype).to(device)
            y_hat_curr_step = (torch.cat([y_q_r for _ in range(8)], dim=1) + means) * mask_list[i]
            y_hat_so_far = y_hat_so_far + y_hat_curr_step
            string_idx += 1
        
        y_hat = y_hat_so_far
    
        x_hat = self.g_s(y_hat).clamp_(0,1)

        return {"x_hat": x_hat}