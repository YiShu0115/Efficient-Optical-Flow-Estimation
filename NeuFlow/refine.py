import torch
from NeuFlow import utils

import torch.nn as nn

from ptlflow.models.sea_raft.layer import ConvNextBlock
from ptlflow.models.sea_raft.update import FlowHead, BasicMotionEncoder

########5、6#########
class LL_ConvNextRefine(nn.Module):
    def __init__(self, context_dim, iter_context_dim, corr_dim, num_blocks=5, hidden_dim=128):
        super().__init__()
        self.encoder = BasicMotionEncoder(corr_channel=corr_dim, dim=hidden_dim)
        self.refine = nn.ModuleList([
            ConvNextBlock(hidden_dim + context_dim + iter_context_dim, iter_context_dim)
            for _ in range(num_blocks)
        ])
        self.flow_head = FlowHead(
            input_dim=iter_context_dim,
            hidden_dim=hidden_dim,
            output_dim=iter_context_dim + 6  # 2 (flow) + 2 (weight) + 2 (log_b)
        )

    def init_bhwd(self, batch_size, height, width, device, amp):
        pass

    def forward(self, corrs, context, hidden, flow):
        motion_features = self.encoder(flow, corrs)
        inp = torch.cat([context, motion_features], dim=1)
        for blk in self.refine:
            hidden = blk(torch.cat([hidden, inp], dim=1))
        flow_update = self.flow_head(hidden)
        delta_flow = flow_update[:, :2]  # [B, 2, H, W]
        info = flow_update[:, 2:6]  # [B, 4, H, W] (weight + log_b)
        hidden = flow_update[:, 6:]  # [B, iter_context_dim, H, W]
        return hidden, delta_flow, info
class ConvNextRefine(nn.Module):
    def __init__(self, context_dim, iter_context_dim, corr_dim, num_blocks=5, hidden_dim=128):
        super().__init__()
        self.encoder = BasicMotionEncoder(corr_channel=corr_dim, dim=hidden_dim)
        self.refine = nn.ModuleList([
            ConvNextBlock(hidden_dim + context_dim + iter_context_dim, iter_context_dim)
            for _ in range(num_blocks)
        ])
        self.flow_head = FlowHead(
            input_dim=iter_context_dim,
            hidden_dim=hidden_dim,
            output_dim=iter_context_dim + 2
        )

    def init_bhwd(self, batch_size, height, width, device, amp):
        pass  # No specific initialization needed for SEARAFT's ConvNextRefine

    def forward(self, corrs, context, hidden, flow):
        motion_features = self.encoder(flow, corrs)
        inp = torch.cat([context, motion_features], dim=1)
        for blk in self.refine:
            hidden = blk(torch.cat([hidden, inp], dim=1))
        flow_update = self.flow_head(hidden)
        delta_flow = flow_update[:, :2]
        hidden = flow_update[:, 2:]
        return hidden, delta_flow

class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='zeros', bias=False)
        self.relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):
        return self.relu(self.conv(x))
##############
class Refine(torch.nn.Module):
    def __init__(self, context_dim, iter_context_dim, num_layers, levels, radius, inter_dim):
        super(Refine, self).__init__()

        self.radius = radius

        self.conv1 = ConvBlock((radius*2+1)**2*levels+context_dim+iter_context_dim+2+1, context_dim+iter_context_dim, kernel_size=3, stride=1, padding=1)

        self.conv2 = ConvBlock(context_dim+iter_context_dim, inter_dim, kernel_size=3, stride=1, padding=1)

        self.conv_layers = torch.nn.ModuleList([ConvBlock(inter_dim, inter_dim, kernel_size=3, stride=1, padding=1)
                                                for i in range(num_layers)])

        self.conv3 = torch.nn.Conv2d(inter_dim, iter_context_dim+2, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True)

        # self.hidden_act = torch.nn.Tanh()
        self.hidden_act = torch.nn.Hardtanh(min_val=-4.0, max_val=4.0)
        # self.hidden_norm = torch.nn.BatchNorm2d(feature_dim)

    def init_bhwd(self, batch_size, height, width, device, amp):
        self.radius_emb = torch.tensor(self.radius, dtype=torch.half if amp else torch.float, device=device).view(1,-1,1,1).expand([batch_size,1,height,width])

    def forward(self, corrs, context, iter_context, flow0):

        x = torch.cat([corrs, context, iter_context, flow0, self.radius_emb], dim=1)

        x = self.conv1(x)

        x = self.conv2(x)

        for layer in self.conv_layers:
            x = layer(x)

        x = self.conv3(x)

        return self.hidden_act(x[:,2:]), x[:,:2]
