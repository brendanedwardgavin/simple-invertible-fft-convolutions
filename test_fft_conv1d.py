import torch
import torch.nn.functional as F
import pytest

from fft_conv1d import fft_conv1d, fft_inv_conv1d

TEST_TOL = 1e-5

@pytest.mark.parametrize("signal_dim", [31, 32, 34])
@pytest.mark.parametrize("kernel_dim", [2, 3, 4, 5, 6])
def test_fft_conv1d(signal_dim, kernel_dim):
    B, C = 1, 1
    N = signal_dim
    K = kernel_dim

    kernel = torch.randn(C, 1, K)
    x = torch.randn(B, C, N)
    
    y_torch = F.conv1d(
        F.pad(x, (K//2+1, K//2+1), mode="circular"), 
        kernel, padding="same")[:,:,K//2+1:-(K//2+1)]
    y_fft = fft_conv1d(x, kernel)
    
    assert torch.norm(y_torch-y_fft)/torch.norm(y_torch) < TEST_TOL
    
@pytest.mark.parametrize("signal_dim", [31, 32, 34])
@pytest.mark.parametrize("kernel_dim", [1, 2, 3, 4, 5, 6])
def test_fft_inv_conv1d(signal_dim, kernel_dim):
    B, C = 1, 1
    N = signal_dim
    K = kernel_dim

    device = torch.device("cuda:0")
    kernel = torch.randn(C, 1, K).to(device)
    x = torch.randn(B, C, N).to(device)
    
    y_torch = F.conv1d(
        F.pad(x, (K//2+1, K//2+1), mode="circular"),
        kernel, padding="same")[:,:,K//2+1:-(K//2+1)]
    y_fft = fft_conv1d(x, kernel)
    
    x_torch_inv = fft_inv_conv1d(y_torch, kernel)
    x_fft_inv = fft_inv_conv1d(y_fft, kernel)
    
    x_fft_res = torch.norm(x_fft_inv-x)/torch.norm(x)
    x_torch_res = torch.norm(x_torch_inv-x)/torch.norm(x)
    
    assert x_fft_res < TEST_TOL
    assert x_torch_res < TEST_TOL