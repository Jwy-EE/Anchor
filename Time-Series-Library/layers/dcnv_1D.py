import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class LayerNorm1d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)
        return x

class DCNv3_1D(nn.Module):

    def __init__(
        self,
        channels=64,
        kernel_size=3,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        offset_scale=1.0,
        act_layer='GELU',
        norm_layer='LN',
        center_feature_scale=False,
        remove_center=False,
    ):
        """
        This is a 1D Deformable Convolution, which is compeltely based on PyTorch built-in functions.
        Key Implementation Details:
        - Uses PyTorch's built-in functions for all operations.
        - Implements offset and mask prediction using linear layers.
        - Uses grid sampling for feature extraction.
        """
        super().__init__()
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

        # 1. 生成 Offset 和 Mask 的投影层
        # 输出通道: group * kernel_size
        # Offset 只需要 1 个维度 (时间轴偏移)
        self.offset_linear = nn.Linear(channels, group * kernel_size * 1) 
        self.mask_linear = nn.Linear(channels, group * kernel_size)

        # 2. 输入输出投影
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)

        # 3. 深度卷积 (用于提取生成 offset 的特征)
        self.dw_conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=channels),
            LayerNorm1d(channels) if norm_layer == 'LN' else nn.Identity(),
            nn.GELU() if act_layer == 'GELU' else nn.Identity()
        )

        self._reset_parameters()
    
    def _reset_parameters(self):  # self must be defined because the function uses self's attributes
        nn.init.constant_(self.offset_linear.weight, 0.)
        nn.init.constant_(self.offset_linear.bias, 0.)
        nn.init.constant_(self.mask_linear.weight, 0.)
        nn.init.constant_(self.mask_linear.bias, 0.)
        nn.init.xavier_uniform_(self.input_proj.weight) # offset & mask
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, x):
            """
            Input: (N, L, C)
            Output: (N, L, C)
            """
            is_channels_first = False
            if x.ndim == 3 and x.shape[1] == self.channels:
                x = x.permute(0, 2, 1) 
                is_channels_first = True
            N, L, C = x.shape 

            # ... (前序代码：投影和计算 offset 保持不变) ...
            # 1. 特征预处理
            x_proj = self.input_proj(x) 
            x_feat = x.permute(0, 2, 1) # (N,C,L)
            x_feat = self.dw_conv(x_feat).permute(0, 2, 1) # (N,L,C)

            # 2. 预测 Offset 和 Mask
            offset = self.offset_linear(x_feat) # () (N,L,group*kernel_size*1)
            mask = self.mask_linear(x_feat).reshape(N, L, self.group, -1) # (N,L,group,kernel_size)
            mask = F.softmax(mask, -1).reshape(N, L, -1) # (N,L,group,softmax(kernel_size)) ---> (N,L,group*softmax(kernel_size))

            # 3. 构建采样网格 (Sampling Grid)
            ref_p = torch.arange(L, dtype=torch.float32, device=x.device).view(1, L)
            dil_grid = torch.linspace(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2, self.kernel_size, device=x.device) * self.dilation
            base_grid = ref_p.view(1, L, 1, 1) + dil_grid.view(1, 1, 1, -1) # (1,L,1,kernel_size)

            offset = offset.view(N, L, self.group, self.kernel_size) # (N,L,group,kernel_size)
            sampling_locations = base_grid + offset * self.offset_scale
            sampling_locations = torch.remainder(sampling_locations, L)
            sampling_locations_norm = 2.0 * sampling_locations / (L - 1) - 1.0


            # 1. 重塑 Input: 将 Group 维度移到 Batch 维度
            # (N, L, C) -> (N, L, g, channels // group) -> (N, g, L, channels // group) -> (N*g, channels // group, 1, L)
            # 这样 grid_sample 就把每个 Group 当作独立的 batch 处理
            x_in = x_proj.view(N, L, self.group, self.group_channels)
            x_in = x_in.permute(0, 2, 3, 1).reshape(N * self.group, self.group_channels, 1, L)


            # 2. 重塑 Grid: 同样将 Group 移到 Batch
            # Grid 原本是 (N, L, g, k) -> 变为 (N*g, 1, L*k, 2)
            # 这里的 2 是因为 grid_sample 需要 (x, y) 坐标，我们补一个 y=0
            sampling_grid = sampling_locations_norm.permute(0, 2, 1, 3) # (N, g, L, k)
            sampling_grid = sampling_grid.reshape(N * self.group, L * self.kernel_size) # (N*g, L*k)
            
            # 构造 (x, 0) 坐标对
            grid_x = sampling_grid.view(N * self.group, 1, -1, 1)
            grid_y = torch.zeros_like(grid_x)
            grid = torch.cat([grid_x, grid_y], dim=-1) # (N*g, 1, L*k, 2)

            # 3. sampling. output: (N*g, channels // group, 1, L*k), the attribute of grid_sample is 
            sampled = F.grid_sample( 
                x_in,   # (N*g, channels // group, 1, L)
                grid,   # (N*g, 1, L*k, 2)
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=True
            )

            # 4. 还原维度
            # Output: (N*g, channels // group, 1, L*k) -> (N, g, channels // group, L, k)
            sampled = sampled.view(N, self.group, self.group_channels, L, self.kernel_size)

            # 现在的维度顺序：(N, group, group_channels, L, kernel_size)
            # 我们需要把它调整为 (N, L, group, kernel_size, group_channels) 以便和 mask 相乘
            sampled = sampled.permute(0, 3, 1, 4, 2) 

            # =======================================================

            # 5. 加权聚合 
            mask = mask.view(N, L, self.group, self.kernel_size, 1)
            output = (sampled * mask).sum(dim=3) # Sum over kernel_size
            output = output.reshape(N, L, C) # Merge group and channels

            output = self.output_proj(output)
            
            # 如果输入是 channels_first 格式，转置回来
            if is_channels_first:
                output = output.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
            
            return output
