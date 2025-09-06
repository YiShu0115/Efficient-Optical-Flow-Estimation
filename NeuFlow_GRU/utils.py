import torch
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
from norm import LayerNorm2d, GroupNorm2d

# def normalize(x):
#     x_min = x.min()
#     return (x - x_min) / (x.max() - x_min)

def coords_grid(b, h, w, device, amp):
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.half if amp else torch.float, device=device), torch.arange(w, dtype=torch.half if amp else torch.float, device=device), indexing='ij')  # [H, W]

    grid = torch.stack([xs, ys], dim=0)  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    return grid

def bilinear_sample(img, sample_coords):

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    return img

def flow_warp(feature, flow):

    b, c, h, w = feature.size()

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature, grid)


class GELU(nn.Module):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)"""

    def __init__(self, inplace: bool = False):
        super(GELU, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.gelu(input)
def get_activation(name):
    if name == "relu":
        return nn.ReLU
    elif name == "gelu":
        return GELU
    elif name == "silu":
        return nn.SiLU
    elif name == "mish":
        return nn.Mish
    elif name == "linear":
        return nn.Identity
    else:
        return None
    

def get_norm(name, affine=False, num_groups=8):
    if name == "group":
        return partial(GroupNorm2d, affine=affine, num_groups=num_groups)
    elif name == "layer":
        return partial(LayerNorm2d, affine=affine)
    # elif name == "batch":
    #     return BatchNorm2d
    else:
        return None