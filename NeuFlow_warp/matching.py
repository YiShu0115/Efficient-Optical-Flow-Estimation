import torch.nn.functional as F

from NeuFlow_warp import utils

import torch
class Matching:

    def init_bhwd(self, batch_size, height, width, device, amp):
        self.grid = utils.coords_grid(batch_size, height, width, device, amp)  # [B, 2, H, W]
        self.flatten_grid = self.grid.view(batch_size, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    def global_correlation_softmax(self, feature0, feature1):

        b, c, h, w = feature0.shape

        feature0 = feature0.flatten(-2).permute(0, 2, 1)
        feature1 = feature1.flatten(-2).permute(0, 2, 1)
        # print(feature0.shape)
        # print(feature1.shape)
        # print(self.flatten_grid.shape)
        # device = torch.device('cuda')
        # self.flatten_grid = torch.rand(1, 1872, 2) .to(device).half()
        correspondence = F.scaled_dot_product_attention(feature0, feature1, self.flatten_grid)

        correspondence = correspondence.view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]
        # print(correspondence.shape)
        # print(self.grid.shape)

        # self.grid = torch.rand(1, 2, 24, 78) .to(device).half()
        flow = correspondence - self.grid

        return flow
