from math import  floor

import torch
import torch.nn.functional as F

def fft_conv1d(x, kernel):
    """
    Simple 1D convolution using FFT that returns same value as 
    torch.functional.conv1d(x, kernel, padding="same") for circular padding of x.
    Assumes 1 channel.
    """
    # channels = 1
    assert kernel.shape[1] == 1 and kernel.shape[0] == 1 
    
    k = kernel.shape[-1]
    
    kernel_padded = F.pad(kernel.flip(2), (0, x.shape[-1]-k), mode="constant")
    kernel_padded = kernel_padded.roll(-floor(k/2), dims=2)
    kernel_fft = torch.fft.fft(kernel_padded)
    x_fft = torch.fft.fft(x)
    y_fft = torch.fft.ifft(kernel_fft * x_fft).real
    return y_fft

def fft_inv_conv1d(x, kernel):
    """
    Simple inverse of 1D convolution using FFT. Assumes 1 channel.
    """
    # channels = 1
    assert kernel.shape[1] == 1 and kernel.shape[0] == 1 
    
    k = kernel.shape[-1]
    kernel_padded = F.pad(kernel.flip(2), (0, x.shape[-1]-k), mode="constant")
    kernel_padded = kernel_padded.roll(-floor(k/2), dims=2)
    kernel_fft = torch.fft.fft(kernel_padded)
    x_fft = torch.fft.fft(x)
    #this is the only line that is different from fft_conv1d:
    y_fft = torch.fft.ifft(x_fft/kernel_fft).real 
    return y_fft