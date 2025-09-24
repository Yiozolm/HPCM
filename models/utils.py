import torch
import torch.nn as nn

from einops import rearrange

import math
import torch.nn.functional as F


def conv3x3(in_ch, out_ch):
    """3x3 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

def conv1x1(in_ch, out_ch):
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
    
def conv2x2_down(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0)

def deconv2x2_up(in_ch, out_ch):
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, output_padding=0, padding=0)

def conv4x4_down(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

def deconv4x4_up(in_ch, out_ch):
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, output_padding=0, padding=1)

def dwconv3x3(ch):
    return nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, groups=ch)

class pconv3x3(nn.Module):
    def __init__(self, N, N1):
        super().__init__()
        self.N = N
        self.N1 = N1
        self.pconv = nn.Conv2d(self.N1, self.N1, 3, 1, 1)
    
    def forward(self, x):
        x1, x2 = torch.split(x, [self.N1, self.N-self.N1], dim=1)
        x1 = self.pconv(x1)
        x = torch.cat((x1, x2), 1)
        return x   
    

class DWConvRB(nn.Module):
    def __init__(self, N=192, mlp_ratio=2, act=nn.LeakyReLU):
        super().__init__()
        middle_ch = N * mlp_ratio
        self.branch = nn.Sequential(
            dwconv3x3(N),
            conv1x1(N, middle_ch),
            act(),
            conv1x1(middle_ch, N),
        )

    def forward(self, x):
        out = x + self.branch(x)
        return out
    
class PConvRB(nn.Module):
    def __init__(self, N=192, partial_ratio=4, mlp_ratio=2, act=nn.LeakyReLU):
        super().__init__()
        N1 = N // partial_ratio
        middle_ch = N * mlp_ratio
        self.branch = nn.Sequential(
            pconv3x3(N, N1),
            conv1x1(N, middle_ch),
            act(inplace=True),
            conv1x1(middle_ch, N),
        )

    def forward(self, x):
        out = x + self.branch(x)
        return out

class y_spatial_prior_s1_s2(nn.Module):
    def __init__(self, M):
        super().__init__()
        
        self.branch_1 = nn.Sequential(
            DWConvRB(M*3),
            DWConvRB(M*3),
        )
        self.branch_2 = nn.Sequential(
            DWConvRB(M*3),
            conv1x1(3*M,2*M),
        )

    def forward(self, x, quant_step):
        return self.branch_2(self.branch_1(x)*quant_step)

class y_spatial_prior_s3(nn.Module):
    def __init__(self, M):
        super().__init__()
        
        self.branch_1 = nn.Sequential(
            DWConvRB(M*3),
            DWConvRB(M*3),
            DWConvRB(M*3),
        )
        self.branch_2 = nn.Sequential(
            DWConvRB(M*3),
            DWConvRB(M*3),
            conv1x1(3*M,2*M),
        )

    def forward(self, x, quant_step):
        return self.branch_2(self.branch_1(x)*quant_step)

class CrossAttentionCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, window_size, kernel_size, num_heads=32):
        super(CrossAttentionCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
        self.conv_q = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv_k = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv_v = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv_out = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

        self.window_size = window_size

    def forward(self, x_t, h_prev):
        x_t_init = x_t
        x_t = rearrange(x_t, 'b c (w1 p1) (w2 p2)  -> (b w1 w2) c p1 p2', p1=self.window_size, p2=self.window_size)
        h_prev = rearrange(h_prev, 'b c (w1 p1) (w2 p2)  -> (b w1 w2) c p1 p2', p1=self.window_size, p2=self.window_size)
        batch_size, C, H, W = x_t.size()
        q = self.conv_q(x_t)
        k = self.conv_k(h_prev)
        v = self.conv_v(h_prev)

        q = q.view(batch_size, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(batch_size, self.num_heads, self.head_dim, H * W)
        v = v.view(batch_size, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        attn_scores = torch.matmul(q, k) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 1, 3, 2).contiguous().view(batch_size, self.hidden_dim, H, W)
        attn_output = rearrange(attn_output, '(b w1 w2) c p1 p2  -> b c (w1 p1) (w2 p2)', w1=x_t_init.shape[2]//self.window_size, w2=x_t_init.shape[3]//self.window_size)

        h_t = attn_output + self.conv_out(x_t_init)

        return h_t