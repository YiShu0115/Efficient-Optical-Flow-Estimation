import torch
import torch.nn.functional as F
import torch.nn as nn

from NeuFlow_warp import backbone_v7
from NeuFlow_warp import transformer
from NeuFlow_warp import matching
from NeuFlow_warp import corr
from NeuFlow_warp import refine
from NeuFlow_warp import upsample
from NeuFlow_warp import config
from NeuFlow_warp import utils


from huggingface_hub import PyTorchModelHubMixin


class NeuFlow(torch.nn.Module,
              PyTorchModelHubMixin,
              repo_url="https://github.com/neufieldrobotics/NeuFlow_v2", license="apache-2.0", pipeline_tag="image-to-image"):
    def __init__(self):
        super(NeuFlow, self).__init__()

        self.backbone = backbone_v7.CNNEncoder(config.feature_dim_s16, config.context_dim_s16, config.feature_dim_s8, config.context_dim_s8)
        
        self.cross_attn_s16 = transformer.FeatureAttention(config.feature_dim_s16+config.context_dim_s16, num_layers=2, ffn=True, ffn_dim_expansion=1, post_norm=True)
        
        self.matching_s16 = matching.Matching()

        #self.flow_attn_s16 = transformer.FlowAttention(config.feature_dim_s16)

        #self.corr_block_s16 = corr.CorrBlock(radius=4, levels=1)
        #self.corr_block_s8 = corr.CorrBlock(radius=4, levels=1)
        
        self.merge_s8 = torch.nn.Sequential(torch.nn.Conv2d(config.feature_dim_s16 + config.feature_dim_s8, config.feature_dim_s8, kernel_size=3, stride=1, padding=1, bias=False),
                                              torch.nn.GELU(),
                                              torch.nn.Conv2d(config.feature_dim_s8, config.feature_dim_s8, kernel_size=3, stride=1, padding=1, bias=False),
                                              torch.nn.BatchNorm2d(config.feature_dim_s8))

        self.context_merge_s8 = torch.nn.Sequential(torch.nn.Conv2d(config.context_dim_s16 + config.context_dim_s8, config.context_dim_s8, kernel_size=3, stride=1, padding=1, bias=False),
                                           torch.nn.GELU(),
                                           torch.nn.Conv2d(config.context_dim_s8, config.context_dim_s8, kernel_size=3, stride=1, padding=1, bias=False),
                                           torch.nn.BatchNorm2d(config.context_dim_s8))
                                           
        
        #self.refine_s16 = refine.Refine(config.context_dim_s16, config.iter_context_dim_s16, num_layers=5, levels=1, radius=4, inter_dim=128)
        #self.refine_s8 = refine.Refine(config.context_dim_s8, config.iter_context_dim_s8, num_layers=5, levels=1, radius=4, inter_dim=96)
        self.refine_s16 = refine.Refine(config.feature_dim_s16,config.context_dim_s16, config.iter_context_dim_s16, num_layers=5, inter_dim=128)
        self.refine_s8 = refine.Refine(config.feature_dim_s8,config.context_dim_s8, config.iter_context_dim_s8, num_layers=5,inter_dim=96)

        self.conv_s8 = backbone_v7.ConvBlock(3, config.feature_dim_s1, kernel_size=8, stride=8, padding=0)
        self.upsample_s8 = upsample.UpSample(config.feature_dim_s1, upsample_factor=8)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def init_bhwd(self, batch_size, height, width, device, amp=True):

        self.backbone.init_bhwd(batch_size*2, height//16, width//16, device, amp) ##batch_size*2

        self.matching_s16.init_bhwd(batch_size, height//16, width//16, device, amp)

        #self.corr_block_s16.init_bhwd(batch_size, height//16, width//16, device, amp)
        #self.corr_block_s8.init_bhwd(batch_size, height//8, width//8, device, amp)

        self.refine_s16.init_bhwd(batch_size, height//16, width//16, device, amp)
        self.refine_s8.init_bhwd(batch_size, height//8, width//8, device, amp)
        
        self.coords_16 = utils.coords_grid(batch_size, height//16, width//16, device, amp)
        self.coords_8 = utils.coords_grid(batch_size, height//8, width//8, device, amp)

        self.init_iter_context_s16 = torch.zeros(batch_size, config.iter_context_dim_s16, height//16, width//16, device=device, dtype=torch.half if amp else torch.float)
        self.init_iter_context_s8 = torch.zeros(batch_size, config.iter_context_dim_s8, height//8, width//8, device=device, dtype=torch.half if amp else torch.float)
   
 
    def split_features(self, features, context_dim, feature_dim):#这里把feature0和feature1拆开

        context, features = torch.split(features, [context_dim, feature_dim], dim=1)

        context, _ = context.chunk(chunks=2, dim=0)
        feature0, feature1 = features.chunk(chunks=2, dim=0)

        return features, torch.relu(context)

    def forward(self, img0,img1, iters_s16=1, iters_s8=2):

        flow_list = []

        img0 /= 255.
        img1 /= 255.
        #img1,img0:->[B,3,W,H]
        #torch.cat([img0, img1], dim=0
        # img0,img1 = imgs.chunk(chunks=2, dim=0)
        # print(img0.shape)
        features_s16, features_s8 = self.backbone(torch.cat([img0, img1], dim=0))#这里吧img0和img1合并了
        #features_s16:-> [2B,C=feature_dim_s16 + context_dim_s16,W=W/16,H=H/16]
        #features_s8:-> [2B,C=feature_dim_s8 + context_dim_s8,W=W/8,H=H/8]

        features_s16 = self.cross_attn_s16(features_s16)#数据增强
        #features_s16:->[2B,C=feature_dim_s16 + context_dim_s16,W=W/16,H=H/16]

        features_s16, context_s16 = self.split_features(features_s16, config.context_dim_s16, config.feature_dim_s16)#用经过transformer增强的1/16feature训练feature和context
        features_s8, context_s8 = self.split_features(features_s8, config.context_dim_s8, config.feature_dim_s8)#用经过CNN的1/8feature训练feature和context
        
        
        feature0_s16, feature1_s16 = features_s16.chunk(chunks=2, dim=0)#区分img0和img1
        
        flow0 = self.matching_s16.global_correlation_softmax(feature0_s16, feature1_s16)#根据feature1/16用self-attention matching的global信息计算flow
        ###flow0 = self.flow_attn_s16(feature0_s16, flow0)#没有用attention
        #flow0 = torch.zeros(N, 2, H//2, W//2).to(img1.device)
        
        
        #corr_pyr_s16 = self.corr_block_s16.init_corr_pyr(feature0_s16, feature1_s16)#计算correlation column pyramid，这里只有一层
        
        iter_context_s16 = self.init_iter_context_s16#初始值为0

        for i in range(iters_s16):
        
            if self.training and i > 0:
                flow0 = flow0.detach()
                ## iter_context_s16 = iter_context_s16.detach()
            
            coords = (self.coords_16 + flow0).detach()
            warp_0 = utils.bilinear_sampler(feature1_s16, coords.permute(0, 2, 3, 1))
            #corrs = self.corr_block_s16(corr_pyr_s16, flow0)#根据attention估计的flow0 warp correlation column pyramid
            
            #iter_flow =  torch.cat([corrs, context_s16, iter_context_s16, flow0], dim=1)
            iter_flow =  torch.cat([warp_0, context_s16, iter_context_s16, flow0], dim=1)
            iter_context_s16, delta_flow = self.refine_s16(iter_flow)

            flow0 = flow0 + delta_flow

            if self.training:
                up_flow0 = F.interpolate(flow0, scale_factor=16, mode='bilinear') * 16#原图scale
                flow_list.append(up_flow0)

        flow0 = F.interpolate(flow0, scale_factor=2, mode='nearest') * 2#flow0插值为1/8scale

        features_s16 = F.interpolate(features_s16, scale_factor=2, mode='nearest')#feature插值为1/8scale

        features_s8 = self.merge_s8(torch.cat([features_s8, features_s16], dim=1))#feature融合1/8scale和1/16scale

        feature0_s8, feature1_s8 = features_s8.chunk(chunks=2, dim=0)#分开img0和img1

        #corr_pyr_s8 = self.corr_block_s8.init_corr_pyr(feature0_s8, feature1_s8)#得到 corr pyramid，这里只有一层

        context_s16 = F.interpolate(context_s16, scale_factor=2, mode='nearest')#context插值为1/8scale

        context_s8 = self.context_merge_s8(torch.cat([context_s8, context_s16], dim=1))#context融合1/8和1/16scale

        iter_context_s8 = self.init_iter_context_s8#初始值为0

        for i in range(iters_s8):

            if self.training and i > 0:
                flow0 = flow0.detach()
                # iter_context_s8 = iter_context_s8.detach()
            
            coords = (self.coords_8 + flow0).detach()
            warp_0 = utils.bilinear_sampler(feature1_s8, coords.permute(0, 2, 3, 1))
            #corrs = self.corr_block_s8(corr_pyr_s8, flow0)#用flow0修正corrs
            
            #iter_flow =  torch.cat([corrs, context_s8, iter_context_s8, flow0], dim=1)
            iter_flow =  torch.cat([warp_0, context_s8, iter_context_s8, flow0], dim=1)
            iter_context_s8, delta_flow = self.refine_s8(iter_flow)#用所有信息得到delta flow

            flow0 = flow0 + delta_flow

            if self.training or i == iters_s8 - 1:

                feature0_s1 = self.conv_s8(img0)
                up_flow0 = self.upsample_s8(feature0_s1, flow0) * 8
                flow_list.append(up_flow0)

        return flow_list
