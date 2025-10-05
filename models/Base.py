import torch
import torch.nn as nn
import math
from compressai.models import CompressionModel
from compressai.ops import quantize_ste
from compressai.entropy_models import GaussianConditional
import scipy
import numpy as np
import math


def get_scale_table(min, max, levels):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class generalnormalcdf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, beta, y):
        ctx.absy = torch.pow(torch.abs(y),beta)
        ctx.beta = beta
        output = (1+torch.sign(y)*torch.special.gammainc(1/beta, ctx.absy))/2
        ctx.device = y.device
        return output

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.absy
        beta = ctx.beta
        y_grad = torch.exp(-y-torch.special.gammaln(1/beta))*beta/2
        return None, grad_output*y_grad

class GGM(GaussianConditional):
    def __init__(self,
        beta: float = 1.5,
        scale_bound: float = 0.12,
        process='y'  
    ):
        super().__init__(scale_bound=scale_bound, scale_table=None)
        assert process in ('z', 'y')
        self.process = process

        # if self.process == 'y':
        #     from .entropy_coders.unbounded_ans import ubransEncoder, ubransDecoder
        #     self.encoder = ubransEncoder()
        #     self.decoder = ubransDecoder()

        if self.scale_table.numel() == 0:
            self.scale_table = get_scale_table(0.12, 64, 60)

        self.register_buffer("beta", torch.Tensor([beta]))

    def _cdf(self, values):
        return generalnormalcdf.apply(self.beta, values)

    def _likelihood(self, inputs, scales, means=None):
        
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs
        scales = self.lower_bound_scale(scales)
        upper = self._cdf((values+half)/scales)
        lower = self._cdf((values-half)/scales)
        likelihood = upper - lower

        return likelihood

    def _standardized_quantile(self, quantile):
        return scipy.stats.gennorm.ppf(quantile, self.beta)

    def update(self):
        if hasattr(self, 'scales_hyper') and self.process == 'z':
            scales = self.scales_hyper.detach().view(-1)
            scales = self.lower_bound_scale(scales)

            tail_mass = 1e-6
            multiplier = -self._standardized_quantile(tail_mass / 2)
            pmf_center = torch.ceil(scales * multiplier).int()
            pmf_length = 2 * pmf_center + 1
            max_length = torch.max(pmf_length).item()

            device = scales.device
            samples = torch.arange(max_length, device=device).int() - pmf_center[:, None]
            samples = samples.float()
            samples_scale = scales.unsqueeze(1)
            samples_scale = samples_scale.float()

            upper = self._cdf((samples+0.5) / samples_scale)
            lower = self._cdf((samples-0.5) / samples_scale)
            pmf = upper - lower

            tail_mass = lower[:, :1] + (1-upper[:,-1:])

            quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
            quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
            self._quantized_cdf = quantized_cdf
            self._offset = -pmf_center
            self._cdf_length = pmf_length + 2
        else:
            if self.scale_table.numel() == 0:
                self.scale_table = get_scale_table(0.12, 64, 60)

            scale_table = self.lower_bound_scale(self.scale_table)

            tail_mass = 1e-6
            multiplier = -self._standardized_quantile(tail_mass / 2)
            pmf_center = torch.ceil(scale_table * multiplier).int()
            pmf_length = 2 * pmf_center + 1
            max_length = torch.max(pmf_length).item()

            device = scale_table.device
            samples = torch.arange(max_length, device=device).int() - pmf_center[:, None]
            samples = samples.float()
            samples_scale = scale_table.unsqueeze(1)
            samples_scale = samples_scale.float()

            upper = self._cdf((samples+0.5) / samples_scale)
            lower = self._cdf((samples-0.5) / samples_scale)
            pmf = upper - lower

            tail_mass = lower[:, :1] + (1-upper[:,-1:])

            quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
            quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
            self._quantized_cdf = quantized_cdf
            self._offset = -pmf_center
            self._cdf_length = pmf_length + 2

        object.__setattr__(self, '_quantized_cdf', self._quantized_cdf)
        object.__setattr__(self, '_offset', self._offset)
        object.__setattr__(self, '_cdf_length', self._cdf_length)

    @staticmethod
    def _build_indexes(size):
        dims: int = len(size)
        N = size[0]
        C = size[1]

        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()

        return indexes.repeat(N, 1, *size[2:])

    def build_indexes(self, scales):
        device = scales.device
        scale_table = self.scale_table[:-1].to(device).view(1,1,1,1,-1)
        scales_expand = scales.unsqueeze(-1)
        indexes = (scales_expand>scale_table).sum(-1)
        return indexes

    def _get_medians(self):
        medians = self.quantiles[:, :, 1:2]
        return medians

    @staticmethod
    def _extend_ndims(tensor, n):
        return tensor.reshape(-1, *([1] * n)) if n > 0 else tensor.reshape(-1)

    
class BaseWrapper(CompressionModel):
    def __init__(self):
        super().__init__()

        self.mask_for_two_part = {}
        self.mask_for_four_part = {}
        self.mask_for_eight_part = {}
        self.mask_for_rec_s2 = {}

        self.means_hyper = nn.Parameter(torch.zeros(1, 256, 1, 1))
        self.scales_hyper = nn.Parameter(torch.ones(1, 256, 1, 1))

        from compressai.entropy_models import EntropyBottleneck
        self.entropy_bottleneck = EntropyBottleneck(256)
        self.gaussian_conditional = GGM(process='y')
        self.gaussian_conditional.scale_table = torch.Tensor() 

    def update(self):
        self.entropy_bottleneck.scales_hyper = self.scales_hyper
        self.entropy_bottleneck.update()
        print('entropy_bottleneck updated')
        self.gaussian_conditional.update()
        print('gaussian_conditional updated')

    def process_with_mask(self, y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask
        y_res = (y - means_hat) * mask
        if self.training:
            y_q = quantize_ste(y_res)
        else:
            y_q = torch.round(y_res)
        y_hat = y_q + means_hat
        return y_res, y_q, y_hat, scales_hat
    
    def get_one_channel_two_parts_mask(self, height, width, dtype, device):
        micro_mask = torch.tensor(((1, 0), (0, 1)), dtype=dtype, device=device)
        mask_0 = micro_mask.repeat(height // 2, width // 2)
        mask_0 = mask_0.unsqueeze(0).unsqueeze(0)
        mask_1 = torch.ones_like(mask_0) - mask_0
        return [mask_0, mask_1]
    
    def get_mask_two_parts(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        with torch.no_grad():
            if curr_mask_str not in self.mask_for_two_part:
                assert channel % 2 == 0
                m = torch.ones((batch, channel // 2, height, width), dtype=dtype, device=device)
                m0, m1 = self.get_one_channel_two_parts_mask(height, width, dtype, device)
                mask_0 = torch.cat((m * m0, m * m1), dim=1)
                mask_1 = torch.cat((m * m1, m * m0), dim=1)
                self.mask_for_two_part[curr_mask_str] = [mask_0, mask_1]
        return self.mask_for_two_part[curr_mask_str]
    
    def get_one_channel_four_parts_mask(self, height, width, dtype, device):
        micro_mask_0 = torch.tensor(((1, 0), (0, 0)), dtype=dtype, device=device)
        mask_0 = micro_mask_0.repeat((height + 1) // 2, (width + 1) // 2)
        mask_0 = mask_0[:height, :width]
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_0 = torch.unsqueeze(mask_0, 0)

        micro_mask_1 = torch.tensor(((0, 1), (0, 0)), dtype=dtype, device=device)
        mask_1 = micro_mask_1.repeat((height + 1) // 2, (width + 1) // 2)
        mask_1 = mask_1[:height, :width]
        mask_1 = torch.unsqueeze(mask_1, 0)
        mask_1 = torch.unsqueeze(mask_1, 0)

        micro_mask_2 = torch.tensor(((0, 0), (1, 0)), dtype=dtype, device=device)
        mask_2 = micro_mask_2.repeat((height + 1) // 2, (width + 1) // 2)
        mask_2 = mask_2[:height, :width]
        mask_2 = torch.unsqueeze(mask_2, 0)
        mask_2 = torch.unsqueeze(mask_2, 0)

        micro_mask_3 = torch.tensor(((0, 0), (0, 1)), dtype=dtype, device=device)
        mask_3 = micro_mask_3.repeat((height + 1) // 2, (width + 1) // 2)
        mask_3 = mask_3[:height, :width]
        mask_3 = torch.unsqueeze(mask_3, 0)
        mask_3 = torch.unsqueeze(mask_3, 0)

        return mask_0, mask_1, mask_2, mask_3

    def get_mask_four_parts(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        with torch.no_grad():
            if curr_mask_str not in self.mask_for_four_part:
                assert channel % 4 == 0
                m = torch.ones((batch, channel // 4, height, width), dtype=dtype, device=device)
                m0, m1, m2, m3 = self.get_one_channel_four_parts_mask(height, width, dtype, device)
                mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
                mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
                mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
                mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)
                self.mask_for_four_part[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.mask_for_four_part[curr_mask_str]
    
    def get_one_channel_eight_parts_mask(self, height, width, dtype, device):
        patten_list = [((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0)), ((0, 0, 1, 0), (0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)), \
                       ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1)), ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0)), \
                       ((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0)), ((0, 0, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0)), \
                       ((0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0)), ((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0), (1, 0, 0, 0))]
        mask_list = []
        for i in range(len(patten_list)):
            micro_mask = torch.tensor(patten_list[i], dtype=dtype, device=device)
            micro_mask = micro_mask.repeat(2, 2)
            mask = micro_mask.repeat((height + 1) // 8, (width + 1) // 8)
            mask = mask[:height, :width]
            mask = torch.unsqueeze(mask, 0)
            mask = torch.unsqueeze(mask, 0)
            mask_list.append(mask)

        return mask_list
    
    def get_one_channel_eight_parts_mask_for_s1(self, height, width, dtype, device):
        patten_list = [((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), ((0, 0, 1, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), \
                       ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0), (0, 0, 0, 0)), \
                       ((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), ((0, 0, 0, 1), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), \
                       ((0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), ((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0), (0, 0, 0, 0))]
        mask_list = []
        for i in range(len(patten_list)):
            micro_mask = torch.tensor(patten_list[i], dtype=dtype, device=device)
            mask = micro_mask.repeat((height + 1) // 4, (width + 1) // 4)
            mask = mask[:height, :width]
            mask = torch.unsqueeze(mask, 0)
            mask = torch.unsqueeze(mask, 0)
            mask_list.append(mask)

        return mask_list
    
    def get_one_channel_eight_parts_mask_for_s2(self, height, width, dtype, device):
        patten_list = [((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0)), ((0, 0, 1, 0), (0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)), \
                       ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1)), ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0)), \
                       ((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0)), ((0, 0, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0)), \
                       ((0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0)), ((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0), (1, 0, 0, 0))]
        mask_list = []
        for i in range(len(patten_list)):
            micro_mask = torch.tensor(patten_list[i], dtype=dtype, device=device)
            mask = micro_mask.repeat((height + 1) // 4, (width + 1) // 4)
            mask = mask[:height, :width]
            mask = torch.unsqueeze(mask, 0)
            mask = torch.unsqueeze(mask, 0)
            mask_list.append(mask)

        return mask_list

    def get_mask_eight_parts(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        with torch.no_grad():
            if curr_mask_str not in self.mask_for_eight_part:
                assert channel % 8 == 0
                mask_list = []
                m = torch.ones((batch, channel // 8, height, width), dtype=dtype, device=device)
                mask_list_one_channel = self.get_one_channel_eight_parts_mask(height, width, dtype, device)
                cat_list = [
                    [0, 2, 4, 6, 1, 3, 5, 7], 
                    [1, 3, 5, 7, 0, 2, 4, 6], 
                    [2, 4, 6, 0, 3, 5, 7, 1], 
                    [3, 5, 7, 1, 2, 4, 6, 0], 
                    [4, 6, 0, 2, 5, 7, 1, 3], 
                    [5, 7, 1, 3, 4, 6, 0, 2], 
                    [6, 0, 2, 4, 7, 1, 3, 5], 
                    [7, 1, 3, 5, 6, 0, 2, 4], 
                ]
                for i in range(8):
                    mask_list.append(torch.cat([m * mask_list_one_channel[cat_list[i][j]] for j in range(8)], dim=1))

                self.mask_for_eight_part[curr_mask_str] = mask_list
        return self.mask_for_eight_part[curr_mask_str]
    
    def get_mask_for_s1(self, batch, channel, height, width, dtype, device):
        assert channel % 8 == 0
        mask_list = []
        m = torch.ones((batch, channel // 8, height, width), dtype=dtype, device=device)
        mask_list_one_channel = self.get_one_channel_eight_parts_mask_for_s1(height, width, dtype, device)
        indices = [0, 2, 4, 6, 1, 3, 5, 7]
        for i in indices:
            mask_list.append(m * mask_list_one_channel[i])

        return mask_list
    
    def get_mask_for_rec_s2(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        with torch.no_grad():
            if curr_mask_str not in self.mask_for_rec_s2:
                assert channel % 4 == 0
                m = torch.ones((batch, channel // 4, height, width), dtype=dtype, device=device)
                m0, m1, m2, m3 = self.get_one_channel_four_parts_mask(height, width, dtype, device)
                mask_0 = m * m0
                mask_1 = m * m1
                mask_2 = m * m2
                mask_3 = m * m3
                self.mask_for_rec_s2[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.mask_for_rec_s2[curr_mask_str]
    
    def get_mask_for_s2(self, batch, channel, height, width, dtype, device):
        assert channel % 8 == 0
        mask_list = []
        m = torch.ones((batch, channel // 8, height, width), dtype=dtype, device=device)
        mask_list_one_channel = self.get_one_channel_eight_parts_mask_for_s2(height, width, dtype, device)

        indices_1 = [0,2,4,6,1,3,5,7]
        indices_2 = [1,3,5,7,0,2,4,6]
        for i in range(8):
            mask_list.append(m * mask_list_one_channel[indices_1[i]] + m * mask_list_one_channel[indices_2[i]])

        return mask_list
    
    def get_s2_hyper_with_mask(self, y, mask_list_s1, mask_list_s2, mask_list_rec_s2, batch, channel, height, width, dtype, device):
        recon_y = torch.zeros((batch, channel, height, width), device=device, dtype=dtype)
        mask = torch.cat([mask_list_s2[i] - mask_list_s1[i] for i in range(len(mask_list_s1))], dim=1)
        mask_rec_s2 = torch.cat(mask_list_rec_s2, dim=1)
        recon_y[~(mask_rec_s2.bool())] = y[mask.bool()]

        return recon_y
    
    def get_s3_hyper_with_mask(self, common_params, mask_list, batch, channel, height, width, dtype, device):
        recon_y = torch.zeros((batch, channel, height, width), device=device, dtype=dtype)
        mask = torch.cat(mask_list, dim=1)
        recon_y[~(mask.bool())] = common_params[~(mask.bool())]

        return recon_y
    
    def get_s1_s2_with_mask(self, y, mask_list, batch, channel, height, width, reduce):
        y_curr_masked_list = []
        slice = channel // reduce
        for i in range(reduce):
            y_curr = y[:, slice * i: slice * (i + 1), :, :]
            y_curr_masked = y_curr.masked_select(mask_list[i].bool()).view(batch, channel // reduce, height, width)
            y_curr_masked_list.append(y_curr_masked)

        return torch.cat(y_curr_masked_list, dim=1)
    
    def recon_for_s2_s3(self, y_curr, mask_list, batch, channel, height, width, dtype, device):
        recon_y = torch.zeros((batch, channel, height, width), device=device, dtype=dtype)
        mask = torch.cat(mask_list, dim=1)
        recon_y[mask.bool()] = y_curr.reshape(-1)
        return recon_y

    @staticmethod
    def combine_for_writing_s1(x):
        return sum(x.chunk(2, 1))
    
    @staticmethod
    def combine_for_writing_s2(x):
        return sum(x.chunk(4, 1))

    @staticmethod
    def combine_for_writing_s3(x):
        return sum(x.chunk(8, 1))