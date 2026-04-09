import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class DSCA(nn.Module):
    def __init__(self, dim, num_heads=4, reduction=4, ste_mask_tau=0.15, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.ste_mask_tau = ste_mask_tau
        self.eps = eps

        hidden_dim = max(dim // 8, 16)
        self.k_dconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.k_mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.q_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.k_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.v_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.head_weights = nn.Parameter(torch.ones(self.num_heads))
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.local_dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.local_pw = nn.Conv2d(dim, dim, 1)
        self.local_act = nn.GELU()

        hidden = max(dim // reduction, 1)
        self.se_reduce = nn.Conv2d(dim, hidden, kernel_size=1, bias=False)
        self.se_expand = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)

        self.fusion = nn.Conv2d(dim * 2, dim, 1)

    def _predict_k(self, x):
        gate = self.k_mlp(self.k_dconv(x))
        return F.adaptive_avg_pool2d(gate, 1).mean(dim=1)

    def _build_hard_mask(self, attn, k_values):
        b = attn.shape[0]
        k_max = min(int(k_values.max().item()), self.head_dim)
        top_k_indices = torch.topk(attn, k=k_max, dim=-1).indices
        arange_k = torch.arange(k_max, device=attn.device).view(1, 1, 1, 1, -1)
        valid = arange_k < k_values.view(b, 1, 1, 1, 1)

        sparse_mask = torch.zeros_like(attn.unsqueeze(2))
        fill_values = torch.ones_like(top_k_indices, dtype=attn.dtype).unsqueeze(2)
        fill_values = fill_values.masked_fill(~valid, 0.0)
        sparse_mask.scatter_(-1, top_k_indices.unsqueeze(2), fill_values)
        return sparse_mask.squeeze(2)

    def _continuous_threshold(self, attn, k_ratio):
        attn_mean = attn.mean(dim=-1, keepdim=True)
        attn_std = attn.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return attn_mean + (1.0 - k_ratio) * attn_std

    def _ste_dcsa(self, x, sparsity_factor):
        b, _, h, w = x.shape
        sparsity_factor = sparsity_factor.clamp(min=self.eps, max=1.0 - self.eps).view(-1)
        k_values_float = 1.0 + (self.head_dim - 1.0) * sparsity_factor
        k_values = torch.floor(k_values_float).long().clamp_(min=1, max=self.head_dim)

        q = self.q_dw(self.q_proj(x))
        k = self.k_dw(self.k_proj(x))
        v = self.v_dw(self.v_proj(x))

        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        soft_prob = F.softmax(attn, dim=-1)
        k_ratio = (k_values_float / self.head_dim).view(b, 1, 1, 1)

        hard_mask = self._build_hard_mask(attn, k_values)
        threshold = self._continuous_threshold(attn, k_ratio)
        soft_mask = torch.sigmoid((attn - threshold) / max(self.ste_mask_tau, self.eps))
        mask = hard_mask + soft_mask - soft_mask.detach()

        prob = soft_prob * mask
        prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        out = prob @ v
        out = out * self.head_weights.view(1, self.num_heads, 1, 1)
        out = out.reshape(b, self.num_heads * self.head_dim, h, w)
        return self.project_out(out)

    def _fgse(self, x):
        _, _, h, w = x.shape
        with torch.no_grad():
            fft_map = torch.fft.rfft2(x, norm='ortho')
            fft_map_abs = torch.abs(fft_map)
            spatial_map = torch.fft.irfft2(fft_map_abs, s=(h, w), norm='ortho')
            spatial_weights = torch.sigmoid(spatial_map.mean(dim=1, keepdim=True))

        squeezed = F.adaptive_avg_pool2d(x * spatial_weights, 1)
        excited = torch.sigmoid(self.se_expand(F.relu(self.se_reduce(squeezed), inplace=True)))
        return x * excited.expand_as(x)

    def forward(self, x):
        sparsity_factor = self._predict_k(x)
        global_out = self._ste_dcsa(x, sparsity_factor)

        local_out = self.local_dw(x)
        local_out = self.local_act(local_out)
        local_out = self._fgse(local_out)
        local_out = self.local_pw(local_out)

        return self.fusion(torch.cat([local_out, global_out], dim=1))


class FEFeedForward(nn.Module):
    def __init__(self, dim, expansion_ratio=1.0, dropout=0.2, bias=False):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        if hidden_dim % 2 != 0:
            hidden_dim += 1
        half_dim = hidden_dim // 2

        self.expand = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=bias)
        self.f5_real = nn.Conv2d(half_dim, half_dim, kernel_size=5, padding=2, groups=half_dim, bias=bias)
        self.f5_imag = nn.Conv2d(half_dim, half_dim, kernel_size=5, padding=2, groups=half_dim, bias=bias)
        self.f3_real = nn.Conv2d(half_dim, half_dim, kernel_size=3, padding=1, groups=half_dim, bias=bias)
        self.f3_imag = nn.Conv2d(half_dim, half_dim, kernel_size=3, padding=1, groups=half_dim, bias=bias)
        self.spatial5 = nn.Conv2d(half_dim, half_dim, kernel_size=5, padding=2, groups=half_dim, bias=bias)
        self.spatial3 = nn.Conv2d(half_dim, half_dim, kernel_size=3, padding=1, groups=half_dim, bias=bias)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.project = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

    @staticmethod
    def _freq_filter(x, conv_r, conv_i):
        xf = torch.fft.fft2(x, norm='ortho')
        return torch.complex(conv_r(xf.real), conv_i(xf.imag))

    @staticmethod
    def _cross_interaction(a, b):
        a1, a2 = torch.chunk(a, 2, dim=1)
        b1, b2 = torch.chunk(b, 2, dim=1)
        return torch.cat([b1, a2], dim=1), torch.cat([a1, b2], dim=1)

    def forward(self, x):
        x0 = self.expand(x)
        x5, x3 = torch.chunk(x0, 2, dim=1)
        f5 = self._freq_filter(x5, self.f5_real, self.f5_imag)
        f3 = self._freq_filter(x3, self.f3_real, self.f3_imag)
        f5, f3 = self._cross_interaction(f5, f3)
        z5 = torch.fft.ifft2(f5, norm='ortho').real
        z3 = torch.fft.ifft2(f3, norm='ortho').real
        u = torch.cat([self.spatial5(z5), self.spatial3(z3)], dim=1)
        u = self.drop1(self.act(u))
        u = self.project(u)
        return self.drop2(u)


class SDABlock(nn.Module):
    def __init__(self, n_feats, num_heads, expansion_ratio=1.0, drop_path=0.1, reduction=4, dropout=0.2, ste_mask_tau=0.15):
        super().__init__()
        self.norm1 = LayerNorm(n_feats)
        self.fused_attn = DSCA(n_feats, num_heads=num_heads, reduction=reduction, ste_mask_tau=ste_mask_tau)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = LayerNorm(n_feats)
        hidden_dim = int(n_feats * expansion_ratio)
        self.ffn_prestep = nn.Sequential(
            nn.Conv2d(n_feats, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, n_feats, 1),
        )
        self.ffn_fe = FEFeedForward(n_feats, expansion_ratio=expansion_ratio, dropout=dropout)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.fused_attn(self.norm1(x)))
        x = x + self.drop_path2(self.ffn_fe(self.ffn_prestep(self.norm2(x))))
        return x


class SDAStage(nn.Module):
    def __init__(self, n_feats, n_block, num_heads, expansion_ratio, drop_path_rate=0.4, reduction=4, dropout=0.2, ste_mask_tau=0.15):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_block)]
        self.body = nn.ModuleList([
            SDABlock(
                n_feats=n_feats,
                num_heads=num_heads,
                expansion_ratio=expansion_ratio,
                drop_path=dpr[i],
                reduction=reduction,
                dropout=dropout,
                ste_mask_tau=ste_mask_tau,
            )
            for i in range(n_block)
        ])
        self.body_tail = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        shortcut = x
        for blk in self.body:
            x = blk(x)
        return self.body_tail(x) + shortcut


class HSI_Upsampler(nn.Module):
    def __init__(self, n_feats, scale):
        super().__init__()
        self.scale = scale

        self.path1_ps = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
        )
        self.path2_interp = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )
        self.fusion = nn.Conv2d(n_feats * 2, n_feats, 1)

    def forward(self, x):
        if self.scale == 4:
            x = self.path1_ps(x)
            x = self.path1_ps(x)
        elif self.scale == 2:
            x = self.fusion(torch.cat([self.path1_ps(x), self.path2_interp(x)], dim=1))
        else:
            for _ in range(int(math.log2(self.scale))):
                x = self.path1_ps(x)
        return x


@ARCH_REGISTRY.register()
class SDANET(nn.Module):
    """
    Due to space limitations in the paper, some basic modules in the code, such as linear layers, may not be explicitly presented. Furthermore, to address the non-differentiability of top‑k selection, we adopt the Straight‑Through Estimator (STE): during the forward pass, a hard top‑k sparse selection is used, while during the backward pass, gradients are passed through a continuous differentiable approximation. This allows us to maintain sparse selection behavior while ensuring stable end‑to‑end training.
    """

    def __init__(
        self,
        n_block=[2, 2, 2, 2],
        n_group=4,
        n_colors=128,
        n_lms_colors=128,
        n_feats=128,
        scale=4,
        num_heads=4,
        expansion_ratio=1.0,
        drop_path_rate=0.4,
        reduction=4,
        dropout=0.2,
        ste_mask_tau=0.15,
    ):
        super().__init__()
        self.scale = scale

        self.head = nn.Conv2d(n_colors, n_feats, 3, 1, 1)
        self.body = nn.ModuleList([
            SDAStage(
                n_feats=n_feats,
                n_block=n_block[i],
                num_heads=num_heads,
                expansion_ratio=expansion_ratio,
                drop_path_rate=drop_path_rate,
                reduction=reduction,
                dropout=dropout,
                ste_mask_tau=ste_mask_tau,
            )
            for i in range(n_group)
        ])
        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

        self.upsampler = HSI_Upsampler(n_feats, scale)
        self.lms_proc = nn.Conv2d(n_lms_colors, n_feats, 3, 1, 1)
        self.final_conv = nn.Conv2d(n_feats, n_colors, 3, 1, 1)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_size = 8
        mod_pad_h = (mod_size - h % mod_size) % mod_size
        mod_pad_w = (mod_size - w % mod_size) % mod_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    def forward_origin(self, x, lms):
        h, w = x.shape[2:]
        x = self.check_image_size(x)

        shortcut = self.head(x)
        res = shortcut
        for group in self.body:
            res = group(res)
        res = self.body_tail(res)
        res = res + shortcut

        upsampled_res = self.upsampler(res)
        target_h, target_w = upsampled_res.shape[2:]
        lms_feat = self.lms_proc(lms[:, :, 0:target_h, 0:target_w])

        output = self.final_conv(upsampled_res + lms_feat)
        return output[:, :, 0:h * self.scale, 0:w * self.scale]

    def forward(self, img_lq, lms):
        return self.forward_origin(img_lq, lms)


