import torch
import torch.nn.functional as F
import pytest

from fft_conv2d import fft_conv2d, fft_inv_conv2d

TEST_TOL = 1e-5

@pytest.mark.parametrize("signal_dim", [(32,32), (31,31), (34,34)])
@pytest.mark.parametrize("kernel_dim", [(3,3), (4,4), (5,5), (6,6)])
def test_fft_conv2d(signal_dim, kernel_dim):
    B, C = 1, 1
    N = signal_dim
    K = kernel_dim

    kernel = torch.randn(C, 1, *K)
    x = torch.randn(B, C, *N)
    
    torch_conv = torch.nn.Conv2d(1, 1, K, padding="same", bias=False, padding_mode='circular')
    torch_conv.weight.data = kernel
    y_torch = torch_conv(x)

    y_fft = fft_conv2d(x, kernel)
    
    relative_residual = torch.norm(y_torch-y_fft)/torch.norm(y_torch)
    assert  relative_residual < TEST_TOL
    
@pytest.mark.parametrize("signal_dim", [(32,32), (31,31), (34,34)])
@pytest.mark.parametrize("kernel_dim", [(3,3), (4,4), (5,5), (6,6)])
def test_fft_inv_conv2d(signal_dim, kernel_dim):
    B, C = 1, 1
    N = signal_dim
    K = kernel_dim

    kernel = torch.randn(C, 1, *K)
    x = torch.randn(B, C, *N)
    
    torch_conv = torch.nn.Conv2d(1, 1, K, padding="same", bias=False, padding_mode='circular')
    torch_conv.weight.data = kernel
    y_torch = torch_conv(x)
    
    y_fft = fft_conv2d(x, kernel)
    
    x_torch_inv = fft_inv_conv2d(y_torch, kernel)
    x_fft_inv = fft_inv_conv2d(y_fft, kernel)
    
    x_fft_res = torch.norm(x_fft_inv-x)/torch.norm(x)
    x_torch_res = torch.norm(x_torch_inv-x)/torch.norm(x)
    
    assert x_fft_res < TEST_TOL
    assert x_torch_res < TEST_TOL