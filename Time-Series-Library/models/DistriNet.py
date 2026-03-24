import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str, target_slice: slice = None) -> torch.Tensor:
        if mode == 'norm':
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x, target_slice)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _get_statistics(self, x: torch.Tensor):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = x.mean(dim=dim2reduce, keepdim=True).detach()
        self.stdev = x.std(dim=dim2reduce, keepdim=True, unbiased=False).detach() + self.eps

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor, target_slice: slice = None) -> torch.Tensor:
        mean = self.mean
        stdev = self.stdev
        weight = self.affine_weight
        bias = self.affine_bias
        if target_slice is not None:
            mean = mean[:, :, target_slice]
            stdev = stdev[:, :, target_slice]
            if self.affine:
                weight = weight[target_slice]
                bias = bias[target_slice]
        if self.affine:
            x = (x - bias) / (weight + self.eps)
        return x * stdev + mean

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, momentum=0.1):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.momentum = momentum
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size + spline_order) * 0.1)
        self.register_buffer('running_min', torch.full((in_features,), -1.0))
        self.register_buffer('running_max', torch.full((in_features,), 1.0))
        grid = self._generate_grid(-1.0, 1.0, in_features, device='cpu')
        self.grid = nn.Parameter(grid, requires_grad=False)
    def _generate_grid(self, x_min, x_max, in_features, device):
        if isinstance(x_min, float):
            x_min = torch.full((in_features,), x_min, device=device)
            x_max = torch.full((in_features,), x_max, device=device)
        h = (x_max - x_min) / self.grid_size
        grid = torch.zeros((in_features, self.grid_size + 2 * self.spline_order + 1), device=device)
        for i in range(in_features):
            grid[i] = torch.linspace(
                x_min[i] - self.spline_order * h[i],
                x_max[i] + self.spline_order * h[i],
                self.grid_size + 2 * self.spline_order + 1,
                device=device
            )
        return grid

    def _auto_update_grid(self, x):
        if not self.training: return
        with torch.no_grad():
            curr_min = x.min(dim=0)[0]
            curr_max = x.max(dim=0)[0]
            self.running_min = (1 - self.momentum) * self.running_min + self.momentum * curr_min
            self.running_max = (1 - self.momentum) * self.running_max + self.momentum * curr_max
            new_grid = self._generate_grid(self.running_min, self.running_max, self.in_features, x.device)
            self.grid.copy_(new_grid)

    def b_splines(self, x):
        x = x.unsqueeze(-1)
        grid = self.grid.unsqueeze(0)
        bases = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).float()
        for k in range(1, self.spline_order + 1):
            left_num = x - grid[:, :, :-(k + 1)]
            left_den = grid[:, :, k:-1] - grid[:, :, :-(k + 1)]
            right_num = grid[:, :, k + 1:] - x
            right_den = grid[:, :, k + 1:] - grid[:, :, 1:-k]
            bases = (left_num / (left_den + 1e-8)) * bases[:, :, :-1] + \
                    (right_num / (right_den + 1e-8)) * bases[:, :, 1:]
        return bases

    def forward(self, x):
        self._auto_update_grid(x)
        base_output = F.linear(x, self.base_weight)
        spline_basis = self.b_splines(x)
        spline_output = torch.einsum("bik,oik->bo", spline_basis, self.spline_weight)
        return base_output + spline_output
class SpectralKANBlock(nn.Module):
    def __init__(self, channels, grid_size=5, freq_groups=4):
        super(SpectralKANBlock, self).__init__()
        self.channels = channels
        self.freq_groups = freq_groups
        self.kan_real = nn.ModuleList([KANLayer(channels, channels, grid_size=grid_size) for _ in range(freq_groups)])
        self.kan_imag = nn.ModuleList([KANLayer(channels, channels, grid_size=grid_size) for _ in range(freq_groups)])
        self.group_weights = nn.Parameter(torch.ones(freq_groups))
    def forward(self, x):
        B, L, C = x.shape
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        real, imag = x_fft.real, x_fft.imag
        V = real.shape[1]
        out_real, out_imag = torch.zeros_like(real), torch.zeros_like(imag)
        for i in range(self.freq_groups):
            start = (i * V) // self.freq_groups
            end = ((i + 1) * V) // self.freq_groups if i < self.freq_groups - 1 else V
            if start == end: continue
            real_group = real[:, start:end, :].reshape(-1, C)
            imag_group = imag[:, start:end, :].reshape(-1, C)
            real_out = self.kan_real[i](real_group).view(B, end-start, C)
            imag_out = self.kan_imag[i](imag_group).view(B, end-start, C)
            out_real[:, start:end, :] = real_out * self.group_weights[i]
            out_imag[:, start:end, :] = imag_out * self.group_weights[i]
        out_fft = torch.complex(out_real, out_imag)
        return torch.fft.irfft(out_fft, n=L, dim=1, norm='ortho')
class MLPLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.c_out = configs.c_out
        self.revin = RevIN(self.channels)
        
        # 核心修改：为每一层分配独立的 LayerNorm 用于 Pre-LN
        self.encoder = nn.ModuleList([
            SpectralKANBlock(
                self.channels, 
                grid_size=getattr(configs, 'grid_size', 5),
                freq_groups=getattr(configs, 'freq_groups', 4)
            ) for _ in range(configs.e_layers)
        ])
        self.encoder_norms = nn.ModuleList([
            nn.LayerNorm(self.channels) for _ in range(configs.e_layers)
        ])
        
        self.target_slice = slice(0, self.c_out)

        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(self.channels, configs.c_out, bias=True)
        elif self.task_name == 'classification':
            self.classification_head = MLPLayer(
                in_features=self.channels,
                hidden_features=getattr(configs, 'd_model', 256),
                out_features=configs.num_class,
                drop=getattr(configs, 'dropout', 0.1)
            )
        else:
            self.projection = nn.Linear(self.channels, configs.c_out, bias=True)

    def _execute_encoder(self, z):
        # 核心逻辑：z = z + layer(norm(z))
        for i, layer in enumerate(self.encoder):
            z = z + layer(self.encoder_norms[i](z))
        return z

    def forecast(self, x_enc):
        z = self.revin(x_enc, 'norm')
        z = self._execute_encoder(z)
        
        z = z.permute(0, 2, 1)
        z = self.predict_linear(z).permute(0, 2, 1)
        
        z = self._execute_encoder(z)
        dec_out = self.projection(z)
        return self.revin(dec_out, 'denorm', target_slice=self.target_slice)

    def imputation(self, x_enc):
        z = self.revin(x_enc, 'norm')
        z = self._execute_encoder(z)
        dec_out = self.projection(z)
        return self.revin(dec_out, 'denorm', target_slice=self.target_slice)

    def anomaly_detection(self, x_enc):
        z = self.revin(x_enc, 'norm')
        z = self._execute_encoder(z)
        dec_out = self.projection(z)
        return self.revin(dec_out, 'denorm', target_slice=self.target_slice)

    def classification(self, x_enc):
        z = self.revin(x_enc, 'norm')
        z = self._execute_encoder(z)
        z = torch.mean(z, dim=1) 
        return self.classification_head(z)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            return self.imputation(x_enc)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc)
        return None