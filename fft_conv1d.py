from math import  floor

import torch
import torch.nn.functional as F

def fft_conv1d(x, kernel):
    """
    Simple 1D convolution using FFT that returns same value as 
    torch.nn.Conv1d(1, 1, kernel_size, padding="same", bias=False, padding_mode='circular')
    """
    # channels = 1
    assert kernel.shape[1] == 1 and kernel.shape[0] == 1 

    k_shape = kernel.shape[2:]
    x_shape = x.shape[2:]
    
    k_padding = (
        0, x_shape[-1]-k_shape[-1]
    )
    k_rolling = (
        -floor(k_shape[0]/2)
    )
    dims = 2
    kernel_padded = kernel.flip(dims)
    kernel_padded = F.pad(kernel_padded, k_padding, mode="constant")
    kernel_padded = kernel_padded.roll(k_rolling, dims=dims)
    
    kernel_fft = torch.fft.fftn(kernel_padded, dim=dims)
    x_fft = torch.fft.fftn(x, dim=dims)
    y_fft = torch.fft.ifftn(kernel_fft * x_fft, dim=dims).real
    return y_fft

def fft_inv_conv1d(x, kernel):
    """
    Simple inverse of 1D convolution using FFT. Inverts convolution performed by
    torch.nn.Conv1d(1, 1, kernel_size, padding="same", bias=False, padding_mode='circular')
    """
    # channels = 1
    assert kernel.shape[1] == 1 and kernel.shape[0] == 1 

    k_shape = kernel.shape[2:]
    x_shape = x.shape[2:]
    
    k_padding = (
        0, x_shape[-1]-k_shape[-1]
    )
    k_rolling = (
        -floor(k_shape[0]/2)
    )
    dims = 2
    kernel_padded = kernel.flip(dims)
    kernel_padded = F.pad(kernel_padded, k_padding, mode="constant")
    kernel_padded = kernel_padded.roll(k_rolling, dims=dims)
    
    kernel_fft = torch.fft.fftn(kernel_padded, dim=dims)
    x_fft = torch.fft.fftn(x, dim=dims)
    #this is the only line that is different from fft_conv1d:
    y_fft = torch.fft.ifftn(x_fft/kernel_fft, dim=dims).real 
    return y_fft