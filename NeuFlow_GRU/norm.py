import torch
import torch.nn as nn

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels of '2D' spatial NCHW tensors"""

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        if self.weight is not None:
            x = F.layer_norm(
                x,
                (x.shape[-1],),
                self.weight,
                self.bias,
                self.eps,
            )
        else:
            x = F.layer_norm(x, (x.shape[-1],), eps=self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
    
class GroupNorm2d(nn.GroupNorm):
    """GroupNorm for channels of '2D' spatial NCHW tensors"""

    def __init__(self, num_groups, num_channels, eps=1e-6, affine=True):
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is not None:
            x = F.group_norm(
                x,
                self.num_groups,
                self.weight,
                self.bias,
                self.eps,
            )
        else:
            x = F.group_norm(x, self.num_groups, eps=self.eps)
        return x