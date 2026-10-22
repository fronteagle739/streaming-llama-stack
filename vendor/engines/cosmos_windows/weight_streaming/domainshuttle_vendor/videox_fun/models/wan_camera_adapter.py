# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/domainshuttle_vendor/videox_fun/models/wan_camera_adapter.py
# Machine fallback paths cleared; no model data included.
import torch
import torch.nn as nn

class SimpleAdapter(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size, stride, downscale_factor=8, num_residual_blocks=1):
        super(SimpleAdapter, self).__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=downscale_factor)
        self.conv = nn.Conv2d(in_dim * downscale_factor * downscale_factor, out_dim, kernel_size=kernel_size, stride=stride, padding=0)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(out_dim) for _ in range(num_residual_blocks)])

    def forward(self, x):
        bs, c, f, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs * f, c, h, w)
        x_unshuffled = self.pixel_unshuffle(x)
        x_conv = self.conv(x_unshuffled)
        out = self.residual_blocks(x_conv)
        out = out.view(bs, f, out.size(1), out.size(2), out.size(3))
        out = out.permute(0, 2, 1, 3, 4)
        return out

class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out
