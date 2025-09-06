import torch
from NeuFlow_GRU.utils import get_norm
import torch.nn as nn
import torch.nn.functional as F
from norm import LayerNorm2d
from cgu import CGUStage




class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='zeros', bias=False)
        self.relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):
        return self.relu(self.conv(x))
    
class CGUGRU(nn.Module):
    def __init__(
        self,
        hidden_dim,
        input_dim,
        activation_function=None,
        norm_layer=LayerNorm2d,
        depth=2,
        mlp_ratio=2,
        mlp_use_dw_conv=True,
        mlp_dw_kernel_size=7,
        mlp_in_kernel_size=1,
        mlp_out_kernel_size=1,
        layer_scale_init_value=1e-2,
    ):
        super(CGUGRU, self).__init__()

        self.convz = CGUStage(
            hidden_dim + input_dim,
            hidden_dim,
            stride=1,
            activation_function=activation_function,
            norm_layer=norm_layer,
            depth=depth,
            use_cross=False,
            mlp_ratio=mlp_ratio,
            mlp_use_dw_conv=mlp_use_dw_conv,
            mlp_dw_kernel_size=mlp_dw_kernel_size,
            mlp_in_kernel_size=mlp_in_kernel_size,
            mlp_out_kernel_size=mlp_out_kernel_size,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.convr = CGUStage(
            hidden_dim + input_dim,
            hidden_dim,
            stride=1,
            activation_function=activation_function,
            norm_layer=norm_layer,
            depth=depth,
            use_cross=False,
            mlp_ratio=mlp_ratio,
            mlp_use_dw_conv=mlp_use_dw_conv,
            mlp_dw_kernel_size=mlp_dw_kernel_size,
            mlp_in_kernel_size=mlp_in_kernel_size,
            mlp_out_kernel_size=mlp_out_kernel_size,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.convq = CGUStage(
            hidden_dim + input_dim,
            hidden_dim,
            stride=1,
            activation_function=activation_function,
            norm_layer=norm_layer,
            depth=depth,
            use_cross=False,
            mlp_ratio=mlp_ratio,
            mlp_use_dw_conv=mlp_use_dw_conv,
            mlp_dw_kernel_size=mlp_dw_kernel_size,
            mlp_in_kernel_size=mlp_in_kernel_size,
            mlp_out_kernel_size=mlp_out_kernel_size,
            layer_scale_init_value=layer_scale_init_value,
        )

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = self.convz(hx)
        z = torch.sigmoid(z)

        r = self.convr(hx)
        r = torch.sigmoid(r)

        q = self.convq(torch.cat([r * h, x], dim=1))
        q = torch.tanh(q)

        h = (1 - z) * h + z * q
        return h
    
class Refine(torch.nn.Module):
    def __init__(self, context_dim, iter_context_dim, num_layers, levels, radius, inter_dim):
        super(Refine, self).__init__()

        self.radius = radius

        self.conv1 = ConvBlock((radius*2+1)**2*levels+context_dim+iter_context_dim+2+1, context_dim+iter_context_dim, kernel_size=3, stride=1, padding=1)

        self.conv2 = ConvBlock(context_dim+iter_context_dim, inter_dim, kernel_size=3, stride=1, padding=1)
        dec_gru_norm_layer = get_norm(
            "layer",
            affine=False,
            num_groups=8,
        )
        self.conv_layers = torch.nn.ModuleList([
            CGUGRU(
                    hidden_dim=inter_dim,
                    input_dim=inter_dim,
                    activation_function="gelu",
                    norm_layer=dec_gru_norm_layer,
                    depth=2,
                    mlp_ratio=2,
                    mlp_use_dw_conv=True,
                    mlp_dw_kernel_size=3,
                    mlp_in_kernel_size=1,
                    mlp_out_kernel_size=1,
                    layer_scale_init_value=0.01,
                )
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
