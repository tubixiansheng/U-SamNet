import math
import os
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

from models.image_encoder import ImageEncoderViT


def _group_count(channels):
    return 4 if channels % 4 == 0 else 1


def _attention_head_count(channels):
    for candidate in [8, 6, 4, 3, 2]:
        if channels % candidate == 0:
            return candidate
    return 1


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            dim_in,
            dim_in,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=dim_in,
            bias=False,
        )
        self.norm_layer = nn.GroupNorm(_group_count(dim_in), dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ContrastAwareCAB(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.project = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x, return_debug=False):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = torch.sqrt(((x - mean) ** 2).mean(dim=(2, 3), keepdim=True) + 1e-6)
        weights = self.project(torch.cat([mean, std], dim=1))
        output = x * weights
        if return_debug:
            return output, {'channel_gate': weights}
        return output


class SAMPriorAdapter(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.down = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(hidden), hidden)
        self.depthwise = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False)
        self.up = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = self.up(F.gelu(self.depthwise(F.gelu(self.norm(self.down(x))))))
        return x + self.scale * residual


class ResidualSEAttention(nn.Module):
    supports_debug = True

    def __init__(self, dim_in, dim_out, reduction=4):
        super().__init__()
        hidden = max(dim_out // reduction, 4)
        self.pre = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(dim_out), dim_out),
            nn.GELU(),
        )
        self.channel_gate = nn.Sequential(
            nn.Conv2d(dim_out, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim_out, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1, groups=_group_count(dim_out), bias=False),
            nn.GELU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=1, bias=False),
        )

    def forward(self, x, return_debug=False):
        x = self.pre(x)
        pooled = x.mean(dim=(2, 3), keepdim=True)
        gate = self.channel_gate(pooled)
        gated = x * gate
        output = self.refine(gated + x)
        if return_debug:
            return output, {
                'channel_gate': gate,
                'gated_response': gated,
                'output_response': output,
            }
        return output


class ResidualCBAMAttention(nn.Module):
    supports_debug = True

    def __init__(self, dim_in, dim_out, reduction=4):
        super().__init__()
        hidden = max(dim_out // reduction, 4)
        self.pre = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(dim_out), dim_out),
            nn.GELU(),
        )
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(dim_out, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim_out, kernel_size=1, bias=True),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1, groups=_group_count(dim_out), bias=False),
            nn.GELU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=1, bias=False),
        )

    def forward(self, x, return_debug=False):
        x = self.pre(x)
        avg_pool = x.mean(dim=(2, 3), keepdim=True)
        max_pool = F.adaptive_max_pool2d(x, 1)
        channel_gate = torch.sigmoid(self.channel_mlp(avg_pool) + self.channel_mlp(max_pool))
        channel_refined = x * channel_gate
        spatial_descriptor = torch.cat([
            channel_refined.mean(dim=1, keepdim=True),
            channel_refined.amax(dim=1, keepdim=True),
        ], dim=1)
        spatial_gate = self.spatial_gate(spatial_descriptor)
        spatial_refined = channel_refined * spatial_gate
        output = self.refine(spatial_refined + x)
        if return_debug:
            return output, {
                'channel_gate': channel_gate,
                'spatial_gate': spatial_gate,
                'channel_refined': channel_refined,
                'output_response': output,
            }
        return output


class ResidualAxialAttention(nn.Module):
    supports_debug = True

    def __init__(self, dim_in, dim_out):
        super().__init__()
        heads = _attention_head_count(dim_out)
        self.pre = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(dim_out), dim_out),
            nn.GELU(),
        )
        self.height_attention = nn.MultiheadAttention(dim_out, heads, batch_first=True)
        self.width_attention = nn.MultiheadAttention(dim_out, heads, batch_first=True)
        self.post_norm = LayerNorm(dim_out, eps=1e-6, data_format='channels_first')
        self.refine = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1, groups=_group_count(dim_out), bias=False),
            nn.GELU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=1, bias=False),
        )

    def forward(self, x, return_debug=False):
        x = self.pre(x)
        residual = x
        bsz, channels, height, width = x.shape

        height_tokens = x.permute(0, 3, 2, 1).reshape(bsz * width, height, channels)
        height_out, height_weights = self.height_attention(height_tokens, height_tokens, height_tokens, need_weights=return_debug)
        height_out = height_out.reshape(bsz, width, height, channels).permute(0, 3, 2, 1)

        width_tokens = x.permute(0, 2, 3, 1).reshape(bsz * height, width, channels)
        width_out, width_weights = self.width_attention(width_tokens, width_tokens, width_tokens, need_weights=return_debug)
        width_out = width_out.reshape(bsz, height, width, channels).permute(0, 3, 1, 2)

        fused = self.post_norm(residual + height_out + width_out)
        output = self.refine(fused)
        if return_debug:
            height_map = height_weights.mean(dim=1).reshape(bsz, width, height).permute(0, 2, 1).unsqueeze(1)
            width_map = width_weights.mean(dim=1).reshape(bsz, height, width).unsqueeze(1)
            return output, {
                'height_attention': height_map,
                'width_attention': width_map,
                'fused_response': fused,
                'output_response': output,
            }
        return output


def build_attention_block(
    attention_module,
    dim_in,
    dim_out,
    cdbha_mode='dynamic',
    cdbha_use_cab=None,
    cdbha_use_boundary_gate=None,
    cdbha_use_dynamic_routing=None,
    cdbha_router_mode='full',
    cdbha_router_temperature=1.0,
    cdbha_router_residual=0.0,
    cdbha_boundary_scale=1.0,
    cdbha_boundary_gate_strength=1.0,
    cdbha_boundary_gate_residual=0.0,
):
    if attention_module == 'cdbha':
        return ContrastAwareDynamicBoundaryHadamardAttention(
            dim_in,
            dim_out,
            mode=cdbha_mode,
            use_cab=cdbha_use_cab,
            use_boundary_gate=cdbha_use_boundary_gate,
            use_dynamic_routing=cdbha_use_dynamic_routing,
            router_mode=cdbha_router_mode,
            router_temperature=cdbha_router_temperature,
            router_residual=cdbha_router_residual,
            boundary_scale=cdbha_boundary_scale,
            boundary_gate_strength=cdbha_boundary_gate_strength,
            boundary_gate_residual=cdbha_boundary_gate_residual,
        )
    if attention_module == 'se':
        return ResidualSEAttention(dim_in, dim_out)
    if attention_module == 'cbam':
        return ResidualCBAMAttention(dim_in, dim_out)
    if attention_module == 'axial':
        return ResidualAxialAttention(dim_in, dim_out)
    raise ValueError(f'Unsupported attention module: {attention_module}')


class ContrastAwareDynamicBoundaryHadamardAttention(nn.Module):
    supports_debug = True

    def __init__(
        self,
        dim_in,
        dim_out,
        x=8,
        y=8,
        reduction=4,
        mode='dynamic',
        use_cab=None,
        use_boundary_gate=None,
        use_dynamic_routing=None,
        router_mode='full',
        router_temperature=1.0,
        router_residual=0.0,
        boundary_scale=1.0,
        boundary_gate_strength=1.0,
        boundary_gate_residual=0.0,
    ):
        super().__init__()
        if dim_in % 4 != 0:
            raise ValueError(f'dim_in must be divisible by 4, got {dim_in}')
        if mode not in {'dynamic', 'static'}:
            raise ValueError(f'Unsupported CDBHA mode: {mode}')
        if router_mode not in {'full', 'lite'}:
            raise ValueError(f'Unsupported CDBHA router mode: {router_mode}')

        c_dim_in = dim_in // 4
        kernel_size = 3
        padding = (kernel_size - 1) // 2
        self.mode = mode
        self.use_cab = True if use_cab is None else use_cab
        self.use_boundary_gate = (mode == 'dynamic') if use_boundary_gate is None else use_boundary_gate
        self.use_dynamic_routing = (mode == 'dynamic') if use_dynamic_routing is None else use_dynamic_routing
        self.router_mode = router_mode
        self.router_temperature = max(float(router_temperature), 1e-3)
        self.router_residual = min(max(float(router_residual), 0.0), 1.0)
        self.boundary_scale = float(boundary_scale)
        self.boundary_gate_strength = max(float(boundary_gate_strength), 0.0)
        self.boundary_gate_residual = min(max(float(boundary_gate_residual), 0.0), 1.0)

        self.params_xy = nn.Parameter(torch.ones(1, c_dim_in, x, y))
        self.params_zx = nn.Parameter(torch.ones(1, 1, c_dim_in, x))
        self.params_zy = nn.Parameter(torch.ones(1, 1, c_dim_in, y))

        self.conv_xy = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=kernel_size, padding=padding, groups=c_dim_in, bias=False),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=1, bias=False),
        )
        self.conv_zx = nn.Sequential(
            nn.Conv1d(c_dim_in, c_dim_in, kernel_size=kernel_size, padding=padding, groups=c_dim_in, bias=False),
            nn.GELU(),
            nn.Conv1d(c_dim_in, c_dim_in, kernel_size=1, bias=False),
        )
        self.conv_zy = nn.Sequential(
            nn.Conv1d(c_dim_in, c_dim_in, kernel_size=kernel_size, padding=padding, groups=c_dim_in, bias=False),
            nn.GELU(),
            nn.Conv1d(c_dim_in, c_dim_in, kernel_size=1, bias=False),
        )
        self.local_branch = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in, bias=False),
        )

        self.cab_xy = ContrastAwareCAB(c_dim_in, reduction=reduction) if self.use_cab else None
        self.cab_zx = ContrastAwareCAB(c_dim_in, reduction=reduction) if self.use_cab else None
        self.cab_zy = ContrastAwareCAB(c_dim_in, reduction=reduction) if self.use_cab else None
        self.cab_local = ContrastAwareCAB(c_dim_in, reduction=reduction) if self.use_cab else None

        if self.use_dynamic_routing:
            router_hidden = max(c_dim_in, 8)
            if self.router_mode == 'full':
                router_in_channels = c_dim_in * 8
                router_out_channels = 4
            else:
                router_in_channels = c_dim_in * 4
                router_out_channels = 2
            self.branch_router = nn.Sequential(
                nn.Conv2d(router_in_channels, router_hidden, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(router_hidden, router_out_channels, kernel_size=1, bias=True),
            )
        else:
            self.branch_router = None

        if self.use_boundary_gate:
            self.boundary_gate = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in, bias=False),
                nn.GELU(),
                nn.Conv2d(dim_in, 1, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            self.boundary_context = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in, bias=False),
                nn.GELU(),
                nn.Conv2d(dim_in, c_dim_in, kernel_size=1, bias=False),
            )
        else:
            self.boundary_gate = None
            self.boundary_context = None

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.fuse_proj = nn.Conv2d(c_dim_in, dim_in, kernel_size=1, bias=False)
        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in, bias=False),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False),
        )

    def forward(self, x, return_debug=False):
        x = self.norm1(x)
        residual = x
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        xy_weights = self.conv_xy(F.interpolate(self.params_xy, size=x1.shape[2:4], mode='bilinear', align_corners=False))
        x1_input = x1 * xy_weights
        x1 = self.cab_xy(x1_input) if self.use_cab else x1_input

        x2_perm = x2.permute(0, 3, 1, 2)
        zx_weights = F.interpolate(self.params_zx, size=x2_perm.shape[2:4], mode='bilinear', align_corners=False).squeeze(0)
        x2_input = (x2_perm * self.conv_zx(zx_weights).unsqueeze(0)).permute(0, 2, 3, 1)
        x2 = self.cab_zx(x2_input) if self.use_cab else x2_input

        x3_perm = x3.permute(0, 2, 1, 3)
        zy_weights = F.interpolate(self.params_zy, size=x3_perm.shape[2:4], mode='bilinear', align_corners=False).squeeze(0)
        x3_input = (x3_perm * self.conv_zy(zy_weights).unsqueeze(0)).permute(0, 2, 1, 3)
        x3 = self.cab_zy(x3_input) if self.use_cab else x3_input

        x4_input = self.local_branch(x4)
        x4 = self.cab_local(x4_input) if self.use_cab else x4_input

        if self.use_dynamic_routing:
            if self.router_mode == 'full':
                descriptors = []
                for branch in (x1, x2, x3, x4):
                    mean = branch.mean(dim=(2, 3), keepdim=True)
                    std = torch.sqrt(((branch - mean) ** 2).mean(dim=(2, 3), keepdim=True) + 1e-6)
                    descriptors.append(torch.cat([mean, std], dim=1))
                router_logits = self.branch_router(torch.cat(descriptors, dim=1)) / self.router_temperature
                routed_weights = F.softmax(router_logits, dim=1)
                uniform_prior = routed_weights.new_full(routed_weights.shape, 0.25)
                branch_weights = (1.0 - self.router_residual) * routed_weights + self.router_residual * uniform_prior

                routed = (
                    x1 * branch_weights[:, 0:1] +
                    x2 * branch_weights[:, 1:2] +
                    x3 * branch_weights[:, 2:3] +
                    x4 * branch_weights[:, 3:4]
                )
            else:
                structured_branch = (x1 + x2 + x3) / 3.0
                router_descriptors = []
                for branch in (structured_branch, x4):
                    mean = branch.mean(dim=(2, 3), keepdim=True)
                    std = torch.sqrt(((branch - mean) ** 2).mean(dim=(2, 3), keepdim=True) + 1e-6)
                    router_descriptors.append(torch.cat([mean, std], dim=1))
                router_logits = self.branch_router(torch.cat(router_descriptors, dim=1)) / self.router_temperature
                coarse_weights = F.softmax(router_logits, dim=1)
                coarse_uniform = coarse_weights.new_full(coarse_weights.shape, 0.5)
                coarse_weights = (1.0 - self.router_residual) * coarse_weights + self.router_residual * coarse_uniform
                structured_weight = coarse_weights[:, 0:1] / 3.0
                branch_weights = torch.cat([structured_weight, structured_weight, structured_weight, coarse_weights[:, 1:2]], dim=1)
                routed = structured_branch * coarse_weights[:, 0:1] + x4 * coarse_weights[:, 1:2]
        else:
            branch_weights = residual.new_full((residual.size(0), 4, 1, 1), 0.25)
            routed = 0.25 * (x1 + x2 + x3 + x4)

        if self.use_boundary_gate:
            boundary_gate = self.boundary_gate(residual)
            effective_boundary_gate = self.boundary_gate_residual + (1.0 - self.boundary_gate_residual) * boundary_gate
            boundary_context = self.boundary_scale * self.boundary_gate_strength * self.boundary_context(residual) * effective_boundary_gate
        else:
            boundary_gate = residual.new_zeros((residual.size(0), 1, residual.size(2), residual.size(3)))
            effective_boundary_gate = boundary_gate
            boundary_context = routed.new_zeros(routed.shape)

        fused = self.fuse_proj(routed + boundary_context)
        fused = self.norm2(fused + residual)
        output = self.ldw(fused)
        if return_debug:
            debug = {
                'xy_response': x1,
                'zx_response': x2,
                'zy_response': x3,
                'local_response': x4,
                'branch_weights': branch_weights,
                'boundary_gate': boundary_gate,
                'effective_boundary_gate': effective_boundary_gate,
                'boundary_context': boundary_context,
                'fused_response': fused,
                'output_response': output,
            }
            return output, debug
        return output


class ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        groups = _group_count(dim_out)
        self.block = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, dim_out),
            nn.GELU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, dim_out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class CrossScalePriorFusion(nn.Module):
    def __init__(self, local_channels, prior_channels, out_channels):
        super().__init__()
        groups = _group_count(out_channels)
        self.local_proj = nn.Conv2d(local_channels, out_channels, kernel_size=1, bias=False)
        self.prior_proj = nn.Conv2d(prior_channels, out_channels, kernel_size=1, bias=False)
        self.interaction_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, local_feat, prior_feat, return_debug=False):
        prior_feat = F.interpolate(prior_feat, size=local_feat.shape[2:], mode='bilinear', align_corners=False)
        local_feat = self.local_proj(local_feat)
        prior_feat = self.prior_proj(prior_feat)
        interaction = self.interaction_proj(local_feat * prior_feat)
        gate = self.gate(torch.cat([local_feat, prior_feat, interaction], dim=1))
        fused = local_feat + gate * prior_feat + (1.0 - gate) * interaction
        output = self.refine(fused)
        if return_debug:
            debug = {
                'gate': gate,
                'prior': prior_feat,
                'local': local_feat,
                'interaction': interaction,
                'output': output,
            }
            return output, debug
        return output


class DiscrepancyRectifiedPriorFusion(nn.Module):
    def __init__(self, local_channels, prior_channels, out_channels, reduction=4, discrepancy_penalty=0.0):
        super().__init__()
        groups = _group_count(out_channels)
        hidden = max(out_channels // reduction, 4)
        self.discrepancy_penalty = float(discrepancy_penalty)
        self.local_proj = nn.Conv2d(local_channels, out_channels, kernel_size=1, bias=False)
        self.prior_proj = nn.Conv2d(prior_channels, out_channels, kernel_size=1, bias=False)
        self.interaction_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.discrepancy_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.confidence_head = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.boundary_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )
        self.channel_gate = nn.Sequential(
            nn.Conv2d(out_channels * 3, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, local_feat, prior_feat, return_debug=False, return_aux=False):
        prior_feat = F.interpolate(prior_feat, size=local_feat.shape[2:], mode='bilinear', align_corners=False)
        local_feat = self.local_proj(local_feat)
        prior_feat = self.prior_proj(prior_feat)

        interaction = self.interaction_proj(local_feat * prior_feat)
        discrepancy = self.discrepancy_proj(torch.abs(local_feat - prior_feat))
        uncertainty = self.uncertainty_head(discrepancy)
        boundary_context = self.boundary_proj(discrepancy)

        channel_stats = torch.cat([
            interaction.mean(dim=(2, 3), keepdim=True),
            discrepancy.mean(dim=(2, 3), keepdim=True),
            torch.sqrt(((discrepancy - discrepancy.mean(dim=(2, 3), keepdim=True)) ** 2).mean(dim=(2, 3), keepdim=True) + 1e-6),
        ], dim=1)
        channel_gate = self.channel_gate(channel_stats)
        rectified_prior = channel_gate * prior_feat

        stacked_context = torch.cat([local_feat, rectified_prior, interaction, discrepancy], dim=1)
        spatial_gate = self.spatial_gate(stacked_context)
        confidence_seed = self.confidence_head(stacked_context)
        if self.discrepancy_penalty > 0.0:
            discrepancy_energy = discrepancy.mean(dim=1, keepdim=True)
            discrepancy_energy = discrepancy_energy / (discrepancy_energy.mean(dim=(2, 3), keepdim=True) + 1e-6)
            discrepancy_attenuation = 1.0 / (1.0 + self.discrepancy_penalty * discrepancy_energy)
        else:
            discrepancy_attenuation = confidence_seed.new_ones(confidence_seed.shape)
        confidence = confidence_seed * (1.0 - uncertainty) * discrepancy_attenuation
        trusted_prior = confidence * rectified_prior
        fallback_interaction = (1.0 - confidence) * interaction
        fused = local_feat + spatial_gate * trusted_prior + (1.0 - spatial_gate) * fallback_interaction - uncertainty * discrepancy + confidence * boundary_context
        output = self.refine(fused)

        aux = {
            'gate': spatial_gate,
            'channel_gate': channel_gate,
            'uncertainty': uncertainty,
            'confidence_seed': confidence_seed,
            'confidence': confidence,
            'discrepancy_attenuation': discrepancy_attenuation,
            'discrepancy': discrepancy,
            'boundary_context': boundary_context,
            'rectified_prior': rectified_prior,
            'local': local_feat,
            'interaction': interaction,
            'trusted_prior': trusted_prior,
            'fallback_interaction': fallback_interaction,
            'output': output,
        }

        if return_debug:
            return output, aux
        if return_aux:
            return output, aux
        return output


class DecoderStage(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        prior_channels,
        use_attention=True,
        fusion_type='default',
        cdbha_mode='dynamic',
        attention_module='cdbha',
        cdbha_use_cab=None,
        cdbha_use_boundary_gate=None,
        cdbha_use_dynamic_routing=None,
        cdbha_router_mode='full',
        cdbha_router_temperature=1.0,
        cdbha_router_residual=0.0,
        cdbha_boundary_scale=1.0,
        cdbha_boundary_gate_strength=1.0,
        cdbha_boundary_gate_residual=0.0,
        drpf_discrepancy_penalty=0.0,
    ):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )
        if fusion_type == 'drpf':
            self.skip_bridge = DiscrepancyRectifiedPriorFusion(
                skip_channels,
                prior_channels,
                skip_channels,
                discrepancy_penalty=drpf_discrepancy_penalty,
            )
        else:
            self.skip_bridge = CrossScalePriorFusion(skip_channels, prior_channels, skip_channels)
        merge_channels = out_channels + skip_channels
        self.merge = build_attention_block(
            attention_module,
            merge_channels,
            out_channels,
            cdbha_mode=cdbha_mode,
            cdbha_use_cab=cdbha_use_cab,
            cdbha_use_boundary_gate=cdbha_use_boundary_gate,
            cdbha_use_dynamic_routing=cdbha_use_dynamic_routing,
            cdbha_router_mode=cdbha_router_mode,
            cdbha_router_temperature=cdbha_router_temperature,
            cdbha_router_residual=cdbha_router_residual,
            cdbha_boundary_scale=cdbha_boundary_scale,
            cdbha_boundary_gate_strength=cdbha_boundary_gate_strength,
            cdbha_boundary_gate_residual=cdbha_boundary_gate_residual,
        ) if use_attention else ConvBlock(merge_channels, out_channels)

    def forward(self, x, skip, prior, return_debug=False, return_aux=False):
        x = self.up(x)
        if return_debug:
            guided_skip, skip_debug = self.skip_bridge(skip, prior, return_debug=True)
            skip_aux = None
        elif return_aux and isinstance(self.skip_bridge, DiscrepancyRectifiedPriorFusion):
            guided_skip, skip_aux = self.skip_bridge(skip, prior, return_aux=True)
            skip_debug = None
        else:
            guided_skip = self.skip_bridge(skip, prior)
            skip_debug = None
            skip_aux = None

        merge_input = torch.cat([x, guided_skip], dim=1)
        if return_debug and getattr(self.merge, 'supports_debug', False):
            x, merge_debug = self.merge(merge_input, return_debug=True)
        else:
            x = self.merge(merge_input)
            merge_debug = None

        if return_debug:
            debug = {'skip_bridge': skip_debug, 'merge': merge_debug, 'output': x, 'guided_skip': guided_skip}
            return x, guided_skip, debug
        if return_aux:
            aux = {'skip_bridge': skip_aux, 'output': x, 'guided_skip': guided_skip}
            return x, guided_skip, aux
        return x, guided_skip


class SharedSAMDecoderCore(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        groups = _group_count(hidden_channels)
        self.context_gate = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.interaction_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )

    def forward(self, local_feat, prior_feat, global_feat, return_debug=False):
        prior_feat = F.interpolate(prior_feat, size=local_feat.shape[2:], mode='bilinear', align_corners=False)
        global_feat = F.interpolate(global_feat, size=local_feat.shape[2:], mode='bilinear', align_corners=False)
        interaction = self.interaction_proj(local_feat * prior_feat)
        gate = self.context_gate(torch.cat([local_feat, prior_feat, global_feat], dim=1))
        fused = local_feat + gate * prior_feat + (1.0 - gate) * global_feat + interaction
        output = self.refine(fused)
        if return_debug:
            return output, {
                'gate': gate,
                'local': local_feat,
                'prior': prior_feat,
                'global': global_feat,
                'interaction': interaction,
                'output': output,
            }
        return output


class USamNet(nn.Module):
    def __init__(
        self,
        num_classes=1,
        input_channels=3,
        c_list=None,
        bridge=True,
        gt_ds=True,
        img_size=256,
        sam_model='b',
        sam_checkpoint=None,
        sam_freeze=True,
        decoder4_3_fusion='drpf',
        cdbha_mode='dynamic',
        attention_module='cdbha',
        cdbha_use_cab=None,
        cdbha_use_boundary_gate=None,
        cdbha_use_dynamic_routing=None,
        cdbha_router_mode='full',
        cdbha_router_temperature=1.0,
        cdbha_router_residual=0.0,
        cdbha_boundary_scale=1.0,
        cdbha_boundary_gate_strength=1.0,
        cdbha_boundary_gate_residual=0.0,
        drpf_discrepancy_penalty=0.0,
        enable_boundary_head=True,
        enable_uncertainty_heads=True,
        use_sam_adapter=True,
        use_shared_sam_decoder=False,
        shared_sam_decoder_dim=64,
        shared_sam_decoder_mode='full',
    ):
        super().__init__()

        if c_list is None:
            c_list = [16, 32, 48, 64, 96, 128]

        self.bridge = bridge
        self.gt_ds = gt_ds
        self.sam_freeze = sam_freeze
        self.enable_boundary_head = enable_boundary_head
        self.enable_uncertainty_heads = enable_uncertainty_heads
        self.use_sam_adapter = use_sam_adapter
        self.use_shared_sam_decoder = use_shared_sam_decoder
        self.shared_sam_decoder_mode = shared_sam_decoder_mode
        self.attention_module = attention_module

        sam_cfg = {
            'h': {'embed_dim': 1280, 'depth': 32, 'num_heads': 16, 'checkpoint': 'models/sam_pretrain/sam_vit_h_4b8939.pth'},
            'l': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'checkpoint': 'models/sam_pretrain/sam_vit_l_0b3195.pth'},
            'b': {'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'checkpoint': 'models/sam_pretrain/sam_vit_b_01ec64.pth'},
        }
        sam = sam_cfg[sam_model]
        checkpoint_path = sam_checkpoint or sam['checkpoint']

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder1 = ConvBlock(input_channels, c_list[0])
        self.encoder2 = ConvBlock(c_list[0], c_list[1])
        self.encoder3 = ConvBlock(c_list[1], c_list[2])
        self.encoder4 = build_attention_block(
            attention_module,
            c_list[2],
            c_list[3],
            cdbha_mode=cdbha_mode,
            cdbha_use_cab=cdbha_use_cab,
            cdbha_use_boundary_gate=cdbha_use_boundary_gate,
            cdbha_use_dynamic_routing=cdbha_use_dynamic_routing,
            cdbha_router_mode=cdbha_router_mode,
            cdbha_router_temperature=cdbha_router_temperature,
            cdbha_router_residual=cdbha_router_residual,
            cdbha_boundary_scale=cdbha_boundary_scale,
            cdbha_boundary_gate_strength=cdbha_boundary_gate_strength,
            cdbha_boundary_gate_residual=cdbha_boundary_gate_residual,
        )
        self.encoder5 = build_attention_block(
            attention_module,
            c_list[3],
            c_list[4],
            cdbha_mode=cdbha_mode,
            cdbha_use_cab=cdbha_use_cab,
            cdbha_use_boundary_gate=cdbha_use_boundary_gate,
            cdbha_use_dynamic_routing=cdbha_use_dynamic_routing,
            cdbha_router_mode=cdbha_router_mode,
            cdbha_router_temperature=cdbha_router_temperature,
            cdbha_router_residual=cdbha_router_residual,
            cdbha_boundary_scale=cdbha_boundary_scale,
            cdbha_boundary_gate_strength=cdbha_boundary_gate_strength,
            cdbha_boundary_gate_residual=cdbha_boundary_gate_residual,
        )
        self.encoder6 = build_attention_block(
            attention_module,
            c_list[4],
            c_list[5],
            cdbha_mode=cdbha_mode,
            cdbha_use_cab=cdbha_use_cab,
            cdbha_use_boundary_gate=cdbha_use_boundary_gate,
            cdbha_use_dynamic_routing=cdbha_use_dynamic_routing,
            cdbha_router_mode=cdbha_router_mode,
            cdbha_router_temperature=cdbha_router_temperature,
            cdbha_router_residual=cdbha_router_residual,
            cdbha_boundary_scale=cdbha_boundary_scale,
            cdbha_boundary_gate_strength=cdbha_boundary_gate_strength,
            cdbha_boundary_gate_residual=cdbha_boundary_gate_residual,
        )

        self.sam_encoder = ImageEncoderViT(
            img_size=img_size,
            in_chans=input_channels,
            embed_dim=sam['embed_dim'],
            depth=sam['depth'],
            patch_size=32,
            num_heads=sam['num_heads'],
            out_chans=c_list[5],
        )
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.sam_encoder.load_state(checkpoint_path)
        else:
            raise FileNotFoundError(
                f'SAM checkpoint not found: {checkpoint_path}. '
                'Download the requested checkpoint before constructing U-SamNet.'
            )

        if self.sam_freeze:
            for parameter in self.sam_encoder.parameters():
                parameter.requires_grad = False

        self.sam_adapter = SAMPriorAdapter(c_list[5]) if use_sam_adapter else nn.Identity()

        self.bottleneck_bridge = CrossScalePriorFusion(c_list[5], c_list[5], c_list[5])
        self.prior_pyramid = nn.ModuleDict({
            'p5': nn.Sequential(nn.Conv2d(c_list[5], c_list[4], kernel_size=1, bias=False), nn.GroupNorm(_group_count(c_list[4]), c_list[4]), nn.GELU()),
            'p4': nn.Sequential(nn.Conv2d(c_list[5], c_list[3], kernel_size=1, bias=False), nn.GroupNorm(_group_count(c_list[3]), c_list[3]), nn.GELU()),
            'p3': nn.Sequential(nn.Conv2d(c_list[5], c_list[2], kernel_size=1, bias=False), nn.GroupNorm(_group_count(c_list[2]), c_list[2]), nn.GELU()),
            'p2': nn.Sequential(nn.Conv2d(c_list[5], c_list[1], kernel_size=1, bias=False), nn.GroupNorm(_group_count(c_list[1]), c_list[1]), nn.GELU()),
            'p1': nn.Sequential(nn.Conv2d(c_list[5], c_list[0], kernel_size=1, bias=False), nn.GroupNorm(_group_count(c_list[0]), c_list[0]), nn.GELU()),
        })

        self.decoder5 = DecoderStage(c_list[5], c_list[4], c_list[4], c_list[4], use_attention=True, fusion_type='default', cdbha_mode=cdbha_mode, attention_module=attention_module, cdbha_use_cab=cdbha_use_cab, cdbha_use_boundary_gate=cdbha_use_boundary_gate, cdbha_use_dynamic_routing=cdbha_use_dynamic_routing, cdbha_router_mode=cdbha_router_mode, cdbha_router_temperature=cdbha_router_temperature, cdbha_router_residual=cdbha_router_residual, cdbha_boundary_scale=cdbha_boundary_scale, cdbha_boundary_gate_strength=cdbha_boundary_gate_strength, cdbha_boundary_gate_residual=cdbha_boundary_gate_residual)
        self.decoder4 = DecoderStage(c_list[4], c_list[3], c_list[3], c_list[3], use_attention=True, fusion_type=decoder4_3_fusion, cdbha_mode=cdbha_mode, attention_module=attention_module, cdbha_use_cab=cdbha_use_cab, cdbha_use_boundary_gate=cdbha_use_boundary_gate, cdbha_use_dynamic_routing=cdbha_use_dynamic_routing, cdbha_router_mode=cdbha_router_mode, cdbha_router_temperature=cdbha_router_temperature, cdbha_router_residual=cdbha_router_residual, cdbha_boundary_scale=cdbha_boundary_scale, cdbha_boundary_gate_strength=cdbha_boundary_gate_strength, cdbha_boundary_gate_residual=cdbha_boundary_gate_residual, drpf_discrepancy_penalty=drpf_discrepancy_penalty)
        self.decoder3 = DecoderStage(c_list[3], c_list[2], c_list[2], c_list[2], use_attention=True, fusion_type=decoder4_3_fusion, cdbha_mode=cdbha_mode, attention_module=attention_module, cdbha_use_cab=cdbha_use_cab, cdbha_use_boundary_gate=cdbha_use_boundary_gate, cdbha_use_dynamic_routing=cdbha_use_dynamic_routing, cdbha_router_mode=cdbha_router_mode, cdbha_router_temperature=cdbha_router_temperature, cdbha_router_residual=cdbha_router_residual, cdbha_boundary_scale=cdbha_boundary_scale, cdbha_boundary_gate_strength=cdbha_boundary_gate_strength, cdbha_boundary_gate_residual=cdbha_boundary_gate_residual, drpf_discrepancy_penalty=drpf_discrepancy_penalty)
        self.decoder2 = DecoderStage(c_list[2], c_list[1], c_list[1], c_list[1], use_attention=False)
        self.decoder1 = DecoderStage(c_list[1], c_list[0], c_list[0], c_list[0], use_attention=False)

        if self.use_shared_sam_decoder:
            if self.shared_sam_decoder_mode not in {'full', 'residual_aux'}:
                raise ValueError(f'Unsupported shared_sam_decoder_mode: {self.shared_sam_decoder_mode}')
            self.shared_sam_decoder = SharedSAMDecoderCore(shared_sam_decoder_dim)
            stage_channels = [c_list[4], c_list[3], c_list[2]] if self.shared_sam_decoder_mode == 'full' else [c_list[4], c_list[3]]
            self.shared_sam_local_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(channels, shared_sam_decoder_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(_group_count(shared_sam_decoder_dim), shared_sam_decoder_dim),
                    nn.GELU(),
                )
                for channels in stage_channels
            ])
            self.shared_sam_prior_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(channels, shared_sam_decoder_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(_group_count(shared_sam_decoder_dim), shared_sam_decoder_dim),
                    nn.GELU(),
                )
                for channels in stage_channels
            ])
            self.shared_sam_global_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(c_list[5], shared_sam_decoder_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(_group_count(shared_sam_decoder_dim), shared_sam_decoder_dim),
                    nn.GELU(),
                )
                for _ in stage_channels
            ])
            self.shared_sam_out_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(shared_sam_decoder_dim, channels, kernel_size=1, bias=False),
                    nn.GroupNorm(_group_count(channels), channels),
                    nn.GELU(),
                )
                for channels in stage_channels
            ])
            if self.shared_sam_decoder_mode == 'full':
                self.shared_sam_fuse_blocks = nn.ModuleList([
                    ConvBlock(channels * 2, channels)
                    for channels in stage_channels
                ])
            else:
                self.shared_sam_residual_gates = nn.ParameterList([
                    nn.Parameter(torch.zeros(1, channels, 1, 1))
                    for channels in stage_channels
                ])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)
        self.boundary_head = None
        if enable_boundary_head:
            self.boundary_head = nn.Sequential(
                nn.Conv2d(c_list[2], c_list[2], kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_group_count(c_list[2]), c_list[2]),
                nn.GELU(),
                nn.Conv2d(c_list[2], num_classes, kernel_size=1),
            )
        self.uncertainty_heads = None
        if enable_uncertainty_heads:
            self.uncertainty_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(c_list[3], c_list[3], kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(_group_count(c_list[3]), c_list[3]),
                    nn.GELU(),
                    nn.Conv2d(c_list[3], num_classes, kernel_size=1),
                ),
                nn.Sequential(
                    nn.Conv2d(c_list[2], c_list[2], kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(_group_count(c_list[2]), c_list[2]),
                    nn.GELU(),
                    nn.Conv2d(c_list[2], num_classes, kernel_size=1),
                ),
            ])

        if self.gt_ds:
            self.gt_heads = nn.ModuleList([
                nn.Conv2d(c_list[4], num_classes, kernel_size=1),
                nn.Conv2d(c_list[3], num_classes, kernel_size=1),
                nn.Conv2d(c_list[2], num_classes, kernel_size=1),
            ])
            self.mask_heads = nn.ModuleList([
                nn.Conv2d(c_list[4], num_classes, kernel_size=1),
                nn.Conv2d(c_list[3], num_classes, kernel_size=1),
                nn.Conv2d(c_list[2], num_classes, kernel_size=1),
            ])

        for module_name, module in self.named_children():
            if module_name != 'sam_encoder':
                module.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv1d):
            n = module.kernel_size[0] * module.out_channels
            module.weight.data.normal_(0, math.sqrt(2.0 / n))
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def _forward_sam(self, x):
        if self.sam_freeze:
            self.sam_encoder.eval()
            with torch.no_grad():
                return self.sam_encoder(x)
        return self.sam_encoder(x)

    def _build_prior(self, sam_feature, key, reference):
        prior = self.prior_pyramid[key](sam_feature)
        return F.interpolate(prior, size=reference.shape[2:], mode='bilinear', align_corners=False)

    def _to_heatmap(self, tensor, input_shape):
        if tensor.dim() == 4:
            tensor = tensor.mean(dim=1, keepdim=True)
        tensor = F.interpolate(tensor, size=input_shape, mode='bilinear', align_corners=False)
        tensor_min = tensor.amin(dim=(2, 3), keepdim=True)
        tensor_max = tensor.amax(dim=(2, 3), keepdim=True)
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-6)
        return tensor

    def _apply_shared_sam_decoder(self, feature, prior, sam_feature, stage_index, return_debug=False):
        local_hidden = self.shared_sam_local_projs[stage_index](feature)
        prior_hidden = self.shared_sam_prior_projs[stage_index](prior)
        global_hidden = self.shared_sam_global_projs[stage_index](sam_feature)
        if return_debug:
            decoded_hidden, debug = self.shared_sam_decoder(local_hidden, prior_hidden, global_hidden, return_debug=True)
        else:
            decoded_hidden = self.shared_sam_decoder(local_hidden, prior_hidden, global_hidden)
            debug = None
        decoded_feature = self.shared_sam_out_projs[stage_index](decoded_hidden)
        if self.shared_sam_decoder_mode == 'full':
            fused_feature = self.shared_sam_fuse_blocks[stage_index](torch.cat([feature, decoded_feature], dim=1))
        else:
            residual_gate = self.shared_sam_residual_gates[stage_index]
            fused_feature = feature + residual_gate * decoded_feature
            if debug is None:
                debug = {}
            debug['residual_gate'] = residual_gate
        if return_debug:
            return fused_feature, decoded_feature, debug
        return fused_feature, decoded_feature

    def _collect_visual_maps(self, input_shape, sam_feature, prior_maps, decoder_debug, encoder_debug, aux_maps, shared_decoder_debug=None):
        visual_maps = OrderedDict()
        visual_maps['sam/prior_global'] = self._to_heatmap(sam_feature, input_shape)
        for name, prior in prior_maps.items():
            if prior is None:
                continue
            visual_maps[f'sam/{name}'] = self._to_heatmap(prior, input_shape)

        for name, debug in encoder_debug.items():
            if debug is None:
                continue
            for key, value in debug.items():
                if value is None:
                    continue
                visual_maps[f'attention/{name}_{key}'] = self._to_heatmap(value, input_shape)

        for name, debug in decoder_debug.items():
            if debug is None:
                continue
            skip_debug = debug.get('skip_bridge') or {}
            for key, value in skip_debug.items():
                if value is None:
                    continue
                visual_maps[f'fusion/{name}_{key}'] = self._to_heatmap(value, input_shape)
            merge_debug = debug.get('merge') or {}
            for key, value in merge_debug.items():
                if value is None:
                    continue
                visual_maps[f'decoder/{name}_{key}'] = self._to_heatmap(value, input_shape)

        for name, value in (aux_maps or {}).items():
            if value is None:
                continue
            visual_maps[f'aux/{name}'] = self._to_heatmap(value, input_shape)

        for name, debug in (shared_decoder_debug or {}).items():
            if debug is None:
                continue
            for key, value in debug.items():
                if value is None:
                    continue
                visual_maps[f'shared_sam/{name}_{key}'] = self._to_heatmap(value, input_shape)

        return visual_maps

    def forward(self, x, visualize=False):
        input_shape = x.shape[2:]
        if x.shape[2] != x.shape[3]:
            raise ValueError('Input height and width must be equal for the SAM encoder.')

        sam_feature = self.sam_adapter(self._forward_sam(x))

        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        if visualize:
            e4, e4_debug = self.encoder4(self.pool(e3), return_debug=True)
            e5, e5_debug = self.encoder5(self.pool(e4), return_debug=True)
            bottleneck_local, e6_debug = self.encoder6(self.pool(e5), return_debug=True)
        else:
            e4 = self.encoder4(self.pool(e3))
            e5 = self.encoder5(self.pool(e4))
            bottleneck_local = self.encoder6(self.pool(e5))
            e4_debug = e5_debug = e6_debug = None

        bottleneck = self.bottleneck_bridge(bottleneck_local, sam_feature) if self.bridge else bottleneck_local

        p5 = self._build_prior(sam_feature, 'p5', e5)
        p4 = self._build_prior(sam_feature, 'p4', e4)
        p3 = self._build_prior(sam_feature, 'p3', e3)
        p2 = self._build_prior(sam_feature, 'p2', e2)
        p1 = self._build_prior(sam_feature, 'p1', e1)
        shared_d5_debug = shared_d4_debug = shared_d3_debug = None

        if visualize:
            d5, guided5, d5_debug = self.decoder5(bottleneck, e5, p5, return_debug=True)
            if self.use_shared_sam_decoder:
                d5, sam_decoded5, shared_d5_debug = self._apply_shared_sam_decoder(d5, p5, sam_feature, 0, return_debug=True)
            else:
                sam_decoded5 = guided5
            d4, guided4, d4_debug = self.decoder4(d5, e4, p4, return_debug=True)
            if self.use_shared_sam_decoder:
                d4, sam_decoded4, shared_d4_debug = self._apply_shared_sam_decoder(d4, p4, sam_feature, 1, return_debug=True)
            else:
                sam_decoded4 = guided4
            d3, guided3, d3_debug = self.decoder3(d4, e3, p3, return_debug=True)
            if self.use_shared_sam_decoder and self.shared_sam_decoder_mode == 'full':
                d3, sam_decoded3, shared_d3_debug = self._apply_shared_sam_decoder(d3, p3, sam_feature, 2, return_debug=True)
            else:
                sam_decoded3 = guided3
            d4_aux = {'skip_bridge': d4_debug.get('skip_bridge') if d4_debug is not None else None}
            d3_aux = {'skip_bridge': d3_debug.get('skip_bridge') if d3_debug is not None else None}
        else:
            d5, guided5 = self.decoder5(bottleneck, e5, p5)
            if self.use_shared_sam_decoder:
                d5, sam_decoded5 = self._apply_shared_sam_decoder(d5, p5, sam_feature, 0)
            else:
                sam_decoded5 = guided5
            d4, guided4, d4_aux = self.decoder4(d5, e4, p4, return_aux=True)
            if self.use_shared_sam_decoder:
                d4, sam_decoded4 = self._apply_shared_sam_decoder(d4, p4, sam_feature, 1)
            else:
                sam_decoded4 = guided4
            d3, guided3, d3_aux = self.decoder3(d4, e3, p3, return_aux=True)
            if self.use_shared_sam_decoder and self.shared_sam_decoder_mode == 'full':
                d3, sam_decoded3 = self._apply_shared_sam_decoder(d3, p3, sam_feature, 2)
            else:
                sam_decoded3 = guided3
            d5_debug = d4_debug = d3_debug = None
        d2, _ = self.decoder2(d3, e2, p2)
        d1, _ = self.decoder1(d2, e1, p1)

        out = torch.sigmoid(self.final(d1))
        boundary = None
        if self.boundary_head is not None:
            boundary = torch.sigmoid(F.interpolate(self.boundary_head(d3), size=input_shape, mode='bilinear', align_corners=False))

        uncertainty = ()
        if self.uncertainty_heads is not None:
            uncertainty = tuple(
                torch.sigmoid(F.interpolate(head(feature), size=input_shape, mode='bilinear', align_corners=False))
                for head, feature in zip(self.uncertainty_heads, [d4, d3])
            )
        confidence = tuple(1.0 - item for item in uncertainty)
        fusion_uncertainty = tuple(
            skip_bridge['uncertainty']
            for item in [d4_aux, d3_aux]
            for skip_bridge in [item.get('skip_bridge')]
            if skip_bridge is not None and 'uncertainty' in skip_bridge
        )
        fusion_confidence = tuple(
            skip_bridge['confidence']
            for item in [d4_aux, d3_aux]
            for skip_bridge in [item.get('skip_bridge')]
            if skip_bridge is not None and 'confidence' in skip_bridge
        )
        extras = {
            'boundary': boundary,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'fusion_uncertainty': fusion_uncertainty,
            'fusion_confidence': fusion_confidence,
        }

        visual_maps = None
        if visualize:
            visual_maps = self._collect_visual_maps(
                input_shape,
                sam_feature,
                {'prior_p5': p5, 'prior_p4': p4, 'prior_p3': p3, 'prior_p2': p2, 'prior_p1': p1},
                {'decoder5': d5_debug, 'decoder4': d4_debug, 'decoder3': d3_debug},
                {'encoder4': e4_debug, 'encoder5': e5_debug, 'encoder6': e6_debug},
                {
                    'boundary': boundary,
                    'uncertainty_d4': uncertainty[0] if len(uncertainty) > 0 else None,
                    'uncertainty_d3': uncertainty[1] if len(uncertainty) > 1 else None,
                    'fusion_confidence_d4': fusion_confidence[0] if len(fusion_confidence) > 0 else None,
                    'fusion_confidence_d3': fusion_confidence[1] if len(fusion_confidence) > 1 else None,
                    'shared_decoder_d5': sam_decoded5 if self.use_shared_sam_decoder else None,
                    'shared_decoder_d4': sam_decoded4 if self.use_shared_sam_decoder else None,
                    'shared_decoder_d3': sam_decoded3 if self.use_shared_sam_decoder and self.shared_sam_decoder_mode == 'full' else None,
                },
                {
                    'decoder5': shared_d5_debug,
                    'decoder4': shared_d4_debug,
                    'decoder3': shared_d3_debug,
                },
            )

        if not self.gt_ds:
            return {'prediction': out, 'extras': extras, 'visual_maps': visual_maps}

        guided_features = [sam_decoded5, sam_decoded4, sam_decoded3] if self.use_shared_sam_decoder and self.shared_sam_decoder_mode == 'full' else [guided5, guided4, guided3]
        gt_pre = []
        masks = []
        for feature, guided, gt_head, mask_head in zip(
            [d5, d4, d3],
            guided_features,
            self.gt_heads,
            self.mask_heads,
        ):
            gt_pre.append(torch.sigmoid(F.interpolate(gt_head(feature), size=input_shape, mode='bilinear', align_corners=False)))
            masks.append(torch.sigmoid(F.interpolate(mask_head(guided), size=input_shape, mode='bilinear', align_corners=False)))

        return {
            'gt_pre': tuple(gt_pre),
            'masks': tuple(masks),
            'prediction': out,
            'extras': extras,
            'visual_maps': visual_maps,
        }
