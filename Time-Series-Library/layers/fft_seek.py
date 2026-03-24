import torch
import torch.fft
import numpy as np
import matplotlib.pyplot as plt

class PeriodEstimator:
    """
    Period estimator based on TimesNet original logic
    Adapted input format: (Batch, Channel, Length) -> Your UniConvNet format
    """
    def __init__(self, top_k=3):
        self.top_k = top_k

    def __call__(self, x):
        """
        Args:
            x: Input data [B, C, L] (Note: TimesNet original is [B, L, C], adapted here)
        Returns:
            periods: List of top-k physical periods [p1, p2, p3]
        """
        # TimesNet original logic replication
        # 1. FFT transform (along the time axis L, i.e., the last dimension)
        xf = torch.fft.rfft(x, dim=-1) 
        
        # 2. Calculate amplitude and average
        # abs(xf): [B, C, L//2+1]
        # mean(0): average over Batch -> [C, L//2+1]
        # mean(0): average over Channel -> [L//2+1] (get global unified spectrum)
        frequency_list = torch.abs(xf).mean(0).mean(0)
        
        # 3. Remove DC component (detrending)
        # TimesNet original code: frequency_list[0] = 0
        frequency_list[0] = 0
        
        # 4. Select Top-K frequencies
        _, top_list = torch.topk(frequency_list, self.top_k)
        
        # 5. Convert to Numpy for easier calculation
        top_list = top_list.detach().cpu().numpy()
        
        # 6. Calculate physical periods
        # Formula: Period = Length // Frequency_Index
        # Your input x length is x.shape[-1]
        L = x.shape[-1]
        periods = L // top_list
        
        # 7. Sorting and deduplication (optional optimization)
        # TimesNet original returns directly, but for better DCN effect, suggest sorting from large to small
        # and remove possible illegal values (e.g., period=0 or 1)
        periods = [int(p) for p in periods]
        periods = sorted(list(set(periods)), reverse=True) # deduplicate and sort descending
        
        return periods