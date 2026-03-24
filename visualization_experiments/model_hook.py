"""
模型 Hook 模块
用于修改 DCNv3_1D 层以暴露内部变量（base_grid, offset, sampling_locations）
"""
import torch
import torch.nn as nn
import sys
import os

# 添加 Time-Series-Library 到路径，以便导入其中的模块
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIB_ROOT = os.path.join(PROJECT_ROOT, 'Time-Series-Library')
if LIB_ROOT not in sys.path:
    sys.path.insert(0, LIB_ROOT)

from layers.dcnv4_1D import DCNv3_1D as OriginalDCNv3_1D


class DCNv3_1DWithHook(OriginalDCNv3_1D):
    """
    带有 Hook 的 DCNv3_1D 层，可以暴露内部变量供可视化使用
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_for_viz = False  # 是否保存可视化数据
        self.viz_data = None       # 保存的可视化数据
        
    def forward(self, x):
        """重写 forward 函数以保存中间变量"""
        is_channels_first = False
        if x.ndim == 3 and x.shape[1] == self.channels:
            x = x.permute(0, 2, 1) 
            is_channels_first = True
        N, L, C = x.shape 

        x_proj = self.input_proj(x) 
        x_feat = x.permute(0, 2, 1) 

        dynamic_padding = ((self.kernel_size - 1) // 2) * self.dilation

        x_feat = torch.nn.functional.conv1d(
            input=x_feat,
            weight=self.dw_conv.weight,  
            bias=self.dw_conv.bias,      
            stride=self.stride,          
            padding=dynamic_padding,     
            dilation=self.dilation,      
            groups=self.channels         
        )

        x_feat = x_feat.permute(0, 2, 1) 

        x_feat_reduced = self.feature_reduce(x_feat)

        offset = self.offset_linear(x_feat_reduced) 
        mask = self.mask_linear(x_feat_reduced).reshape(N, L, self.group, -1) 

        ref_p = torch.arange(L, dtype=torch.float32, device=x.device).view(1, L)
        dil_grid = torch.linspace(-(self.kernel_size - 1) // 2, 
                                 (self.kernel_size - 1) // 2, 
                                 self.kernel_size, device=x.device) * self.dilation
        base_grid = ref_p.view(1, L, 1, 1) + dil_grid.view(1, 1, 1, -1) 

        offset = offset.view(N, L, self.group, self.kernel_size) 
        sampling_locations = base_grid + offset * self.offset_scale
        sampling_locations = torch.remainder(sampling_locations, L)
        sampling_locations_norm = 2.0 * sampling_locations / (L - 1) - 1.0

        x_in = x_proj.view(N, L, self.group, self.group_channels)
        x_in = x_in.permute(0, 2, 3, 1).reshape(N * self.group, self.group_channels, 1, L)

        sampling_grid = sampling_locations_norm.permute(0, 2, 1, 3) 
        sampling_grid = sampling_grid.reshape(N * self.group, L * self.kernel_size) 
        
        grid_x = sampling_grid.view(N * self.group, 1, -1, 1)
        grid_y = torch.zeros_like(grid_x)
        grid = torch.cat([grid_x, grid_y], dim=-1) 

        with torch.backends.cudnn.flags(enabled=False):
            sampled = torch.nn.functional.grid_sample( 
                x_in,   
                grid,   
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=True
            )

        sampled = sampled.view(N, self.group, self.group_channels, L, self.kernel_size)
        sampled = sampled.permute(0, 3, 1, 4, 2) 

        mask = mask.view(N, L, self.group, self.kernel_size, 1)
        output = (sampled * mask).sum(dim=3) 
        output = output.reshape(N, L, C) 

        output = self.output_proj(output)
        
        if is_channels_first:
            output = output.permute(0, 2, 1)  
        
        # 保存可视化数据
        if self.save_for_viz:
            self.viz_data = {
                'base_grid': base_grid.detach().cpu(),      # shape: (1, L, 1, kernel_size)
                'offset': offset.detach().cpu(),            # shape: (1, L, group, kernel_size)
                'sampling_locations': sampling_locations.detach().cpu(),  # shape: (1, L, group, kernel_size)
                'dilation': self.dilation,
                'kernel_size': self.kernel_size,
                'group': self.group,
                'L': L
            }
        
        return output


def patch_model_with_hooks(model):
    """
    将模型中的所有 DCNv3_1D 层替换为带 Hook 的版本
    
    Args:
        model: ANCHOR 模型实例
    
    Returns:
        修补后的模型
    """
    def replace_layers(module):
        for name, child in module.named_children():
            if isinstance(child, OriginalDCNv3_1D):
                # 获取原始层的设备
                device = next(child.parameters()).device
                
                # 创建新的带 Hook 的层，复制所有参数
                new_layer = DCNv3_1DWithHook(
                    channels=child.channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    pad=child.pad,
                    dilation=child.dilation,
                    group=child.group,
                    offset_scale=child.offset_scale,
                    act_layer='GELU',  # 默认值
                    norm_layer='LN',   # 默认值
                    center_feature_scale=False,
                    remove_center=False
                )
                
                # 将新层移动到与原始层相同的设备
                new_layer = new_layer.to(device)
                
                # 复制权重
                new_layer.load_state_dict(child.state_dict())
                
                # 替换层
                setattr(module, name, new_layer)
                print(f"[Hook Patch] 已将层 {name} 替换为带 Hook 的版本，设备: {device}")
            else:
                # 递归处理子模块
                replace_layers(child)
    
    replace_layers(model)
    return model


def get_layer_by_path(model, layer_path):
    """
    根据路径获取模型中的层
    
    Args:
        model: 模型
        layer_path: 层路径，如 'stages.0.0.rfa.a1.2'
    
    Returns:
        层对象
    """
    parts = layer_path.split('.')
    current = model
    
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    
    return current


def enable_hook_for_layer(model, layer_path, enable=True):
    """
    启用或禁用指定层的 Hook
    
    Args:
        model: 模型
        layer_path: 层路径
        enable: 是否启用 Hook
    """
    layer = get_layer_by_path(model, layer_path)
    if hasattr(layer, 'save_for_viz'):
        layer.save_for_viz = enable
        if not enable:
            layer.viz_data = None
    else:
        print(f"警告: 层 {layer_path} 不是 DCNv3_1DWithHook 类型")


def get_viz_data_from_layer(model, layer_path):
    """
    从指定层获取可视化数据
    
    Args:
        model: 模型
        layer_path: 层路径
    
    Returns:
        可视化数据字典，如果未启用 Hook 则返回 None
    """
    layer = get_layer_by_path(model, layer_path)
    if hasattr(layer, 'viz_data') and layer.viz_data is not None:
        return layer.viz_data
    return None


def clear_all_hooks(model):
    """清除所有层的 Hook 数据"""
    def clear_layer(module):
        if hasattr(module, 'save_for_viz'):
            module.save_for_viz = False
            module.viz_data = None
        for child in module.children():
            clear_layer(child)
    
    clear_layer(model)


if __name__ == "__main__":
    # 测试 Hook 功能
    print("DCNv3_1DWithHook 类已定义")
    print("可以使用 patch_model_with_hooks() 修补模型")
    print("使用 enable_hook_for_layer() 启用特定层的 Hook")
    print("使用 get_viz_data_from_layer() 获取可视化数据")