import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings
try:
    from . import dcnv4_1d_cuda_backend as backend
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    warnings.warn("DCNv4 CUDA extension not compiled, will use PyTorch implementation")
class DCNv4_1D_CUDA(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        kernel_size: int = 3,
        stride: int = 1,
        pad: int = 1,
        dilation: int = 1,
        group: int = None,  
        offset_scale: float = 1.0,
        act_layer: str = 'GELU',
        norm_layer: str = 'LN',
        center_feature_scale: bool = False,
        remove_center: bool = False,
        use_cuda: bool = True,
        use_fused: bool = False,
    ):
        super().__init__()
        if group is None:
            group = max(1, channels // 32)
            if channels % group != 0:
                group = 1  
        if channels % group != 0:
            raise ValueError(f'channels must be divisible by group, but got {channels} and {group}')
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.remove_center = int(remove_center)
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        self.offset_mask_linear = nn.Linear(channels, group * kernel_size * 2)
        self.norm = self._build_norm_layer(norm_layer, channels)
        self.act = self._build_activation(act_layer)
        self._init_fallback()
        self._reset_parameters()
    def _build_norm_layer(self, norm_layer: str, channels: int) -> nn.Module:
        if norm_layer == 'LN':
            class LayerNorm1d(nn.Module):
                def __init__(self, normalized_shape, eps=1e-6):
                    super().__init__()
                    self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)
                def forward(self, x):
                    x = x.permute(0, 2, 1)
                    x = self.layer_norm(x)
                    x = x.permute(0, 2, 1)
                    return x
            return LayerNorm1d(channels)
        else:
            return nn.Identity()
    def _build_activation(self, act_layer: str) -> nn.Module:
        if act_layer == 'GELU':
            return nn.GELU()
        elif act_layer == 'ReLU':
            return nn.ReLU()
        else:
            return nn.Identity()
    def _init_fallback(self):
        from layers.dcnv4_1D import DCNv3_1D
        self.fallback = DCNv3_1D(
            channels=self.channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            pad=self.pad,
            dilation=self.dilation,
            group=self.group,
            offset_scale=self.offset_scale,
            act_layer='GELU',
            norm_layer='LN'
        )
        self._copy_weights_to_fallback()
    def _copy_weights_to_fallback(self):
        pass
    def _reset_parameters(self):
        nn.init.constant_(self.offset_mask_linear.weight, 0.)
        nn.init.constant_(self.offset_mask_linear.bias, 0.)
    def _cuda_forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_cuda or not x.is_cuda:
            return self._pytorch_forward(x)
        try:
            N, C, L = x.shape
            x_norm = self.norm(x)
            x_act = self.act(x_norm)
            x_feat = x_act.permute(0, 2, 1)  
            offset_mask = self.offset_mask_linear(x_feat)  
            total_elements = self.group * self.kernel_size
            offset = offset_mask[..., :total_elements]  
            mask = offset_mask[..., total_elements:]    
            output = backend.dcnv4_forward(
                x_act, offset, mask,
                self.kernel_size, self.group, self.offset_scale,
                self.dilation
            )
            return output
        except Exception as e:
            warnings.warn(f"CUDA forward failed, fallback to PyTorch: {e}")
            return self._pytorch_forward(x)
    def _pytorch_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fallback(x)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and x.shape[1] != self.channels:
            x = x.permute(0, 2, 1)
        if self.use_cuda and x.is_cuda:
            return self._cuda_forward(x)
        else:
            return self._pytorch_forward(x)
    def extra_repr(self) -> str:
        return (f'channels={self.channels}, kernel_size={self.kernel_size}, '
                f'group={self.group}, dilation={self.dilation}, '
                f'use_cuda={self.use_cuda and CUDA_AVAILABLE}')
def create_dcnv4_1d(
    channels: int = 64,
    kernel_size: int = 3,
    use_cuda: bool = True,
    **kwargs
) -> nn.Module:
    return DCNv4_1D_CUDA(
        channels=channels,
        kernel_size=kernel_size,
        use_cuda=use_cuda,
        **kwargs
    )
