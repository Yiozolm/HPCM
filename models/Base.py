import torch

from compressai.models import CompressionModel
from compressai.ops import quantize_ste


class BaseWrapper(CompressionModel):
    def __init__(self):
        super().__init__()

        self.mask_for_two_part = {}
        self.mask_for_four_part = {}
        self.mask_for_eight_part = {}
        self.mask_for_rec_s2 = {}
    

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