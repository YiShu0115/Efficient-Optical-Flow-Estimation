import torch
import torch.nn as nn
from torch.nn import functional as F
import thop
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


import torch
from glob import glob
import os
import numpy as np
import cv2
from NeuFlow.neuflow import NeuFlow
from programming.backbone_v7 import ConvBlock
from data_utils import flow_viz
from PIL import Image




def export_onnx(model, inputs, inputs_names, outputs_names, model_file_name):
    import onnx
    from onnxsim import simplify
    torch.onnx.export(model,                     # model being run
                    inputs,                      # model input (or a tuple for multiple inputs)
                    model_file_name,             # where to save the model (can be a file or file-like object)
                    input_names=inputs_names,
                    output_names=outputs_names,
                    export_params=True,          # store the trained parameter weights inside the model file
                    opset_version=16,            # the ONNX version to export the model to
                    # do_constant_folding=True
                    )    # whether to execute constant folding for optimization
    onnx_model = onnx.load(model_file_name)      # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    print(f"{model_file_name} onnx sim sucess...")
    onnx.save(model_simp, model_file_name)

def split_features(features, feature_dim=128, context_relu=False):
    n, c, h, w = features.shape
    features, context = torch.split(features, [feature_dim, c-feature_dim], dim=1)
    if context_relu:
        context = torch.relu(context)
    return features, context


def fuse_conv_and_bn(conv, bn):
        """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
        fusedconv = (
            torch.nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=True,
            )
            .requires_grad_(False)
            .to(conv.weight.device)
        )

        # Prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        # Prepare spatial bias
        b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv
    
    

import torch
w, h = 768, 432
# w, h = (w//32 + 1)*32, (h//32+1)*32
x1 = torch.randn((1, 3, h, w)).cuda().half()
x2 = torch.randn((1, 3, h, w)).cuda().half()
gt = torch.randn((1, 2, h, w)).cuda().half()
valid = torch.randn((1, h, w)).cuda().half()

if __name__ == "__main__":
    device = torch.device('cuda')
    model = NeuFlow.from_pretrained("./neuflow_things.pth").to(device)


    for m in model.modules():
        if type(m) is ConvBlock:
            m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)  # update conv
            m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)  # update conv
            delattr(m, "norm1")  # remove batchnorm
            delattr(m, "norm2")  # remove batchnorm
            m.forward = m.forward_fuse  # update forward

    model.eval()
    model.half()
    
    model.init_bhwd(1, h, w, device)

     
    y = model(x1, x2)
    Flops, params = thop.profile(model, inputs=(x1, x2)) # macs
    print(f'name: neuflow')
    print('Flops: % .4fG'%(Flops / 1000000000))                            
    print('params: % .4fM'% (params / 1000000))                       
#     model.forward = model.infer
#     export_onnx(model, (x1, x2, ), inputs_names=["img1", "img2"], outputs_names=['flow'], model_file_name=f'onnx/sea_raft_s1_{w}x{h}.onnx')
