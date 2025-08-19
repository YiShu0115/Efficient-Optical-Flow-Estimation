import torch.nn as nn
import torch_pruning as tp
from NeuFlow.neuflow import NeuFlow
import argparse
import torch
from data_utils import datasets
import numpy as np
from loss import flow_loss_func
import copy
import os
from data_prune import MyFlowDataset

from lightning.pytorch import seed_everything
 
seed_everything(3407, workers=True)

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--iter', default=5, type=str,
                        help='number of iterations for pruning')
    parser.add_argument('--pruning_ratio', default=0.5, type=float,
                        help='pruning ratio for each iteration')
    parser.add_argument('--dataset', default='kitti', type=str,
                        help='dataset used for the model')
    parser.add_argument('--resume', default='neuflow_things.pth', type=str,
                    help='resume from pretrain model for finetuing or resume from terminated training')
    return parser
parser = get_args_parser()
args = parser.parse_args()



class SingleInputBackbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, img):
        # 只输入 img1 用于建图（build_dependency）
        return self.backbone(img)
        

def structured_prune_model(model, image1, image2, device, prune_ratio=0.01):
    # 0. 准备工作
    model.eval()
    dummy_input = torch.cat([image1, image2], dim=0)
    dummy_input_refine1 = torch.rand(1, 211, 23, 48) .to(device).half()
    dummy_input_refine2 = torch.rand(1, 211, 46, 96) .to(device).half()
    
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, (image1, image2))
    imp = tp.importance.GroupMagnitudeImportance(p=2) 
    
    # 1. 构建依赖图
    backbone = SingleInputBackbone(model.backbone).to(device)
    
    # 2. Initialize a pruner with the model and the importance criterion
    ignored_layers = []
    for name, m in backbone.named_modules():
        # 忽略名称包含'cat'的层
        if 'cat' in name:
            ignored_layers.append(m)
            
    
    pruner_backbone = tp.pruner.BasePruner( # We can always choose BasePruner if sparse training is not required.
        backbone,
        dummy_input,
        importance=imp,
        pruning_ratio=prune_ratio, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
        ignored_layers=ignored_layers,
        round_to=8, # It's recommended to round dims/channels to 4x or 8x for acceleration. Please see: https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
    )

    # 3. Prune the model
    
    print("=== 剪 backbone 前 ===")
    tp.utils.print_tool.before_pruning(backbone)
    pruner_backbone.step()
    print("=== 剪 backbone 后 ===")
    tp.utils.print_tool.after_pruning(backbone)
    
    model.backbone = backbone.backbone
    
    
    # refine_1 = model.refine_s16.to(device)  # 需要你实现 SingleInputRefine 来适配 refine 的输入
    # pruner_refine = tp.pruner.BasePruner(
    #     refine_1,
    #     dummy_input_refine1.unsqueeze(dim=0),
    #     importance=imp,
    #     pruning_ratio=prune_ratio,
    #     # ignored_layers=ignored_layers_refine,
    #     round_to=8,
    # )

    # print("=== 剪 refine 前 ===")
    # tp.utils.print_tool.before_pruning(refine_1)
    # pruner_refine.step()
    # print("=== 剪 refine 后 ===")
    # tp.utils.print_tool.after_pruning(refine_1)

    # # 把剪过的 refine 挂回原模型
    # model.refine_s16 = refine_1
    
    # refine_2 = model.refine_s8.to(device)  # 需要你实现 SingleInputRefine 来适配 refine 的输入
    # pruner_refine = tp.pruner.BasePruner(
    #     refine_2,
    #     dummy_input_refine2.unsqueeze(dim=0),
    #     importance=imp,
    #     pruning_ratio=prune_ratio,
    #     # ignored_layers=ignored_layers_refine,
    #     round_to=8,
    # )

    # print("=== 剪 refine 前 ===")
    # tp.utils.print_tool.before_pruning(refine_2)
    # pruner_refine.step()
    # print("=== 剪 refine 后 ===")
    # tp.utils.print_tool.after_pruning(refine_2)

    # # 把剪过的 refine 挂回原模型
    # model.refine_s8 = refine_2
    
    macs, nparams = tp.utils.count_ops_and_params(model, (image1, image2))
    print(f"MACs: {base_macs/1e9} G -> {macs/1e9} G, #Params: {base_nparams/1e6} M -> {nparams/1e6} M")
    
    return model  # 返回原始状态备用


def count_active_params(model):
    """计算模型中活跃参数的总数（非零参数）"""
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:  # 只计算可训练参数
            total_params += param.data.count_nonzero().item()
    return total_params
def count_active_params_all(model):
    """计算模型的总参数量（不考虑稀疏性）"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import torch.optim as optim
from tqdm import tqdm  # 进度条工具

def finetune(model, dataloader, device,i, epochs=5):
    torch.backends.cudnn.benchmark = True
    model = model.float()
    """
    简单的微调函数
    Args:
        model: 剪枝后的模型
        dataloader: 数据加载器
        device: 设备 (cuda/cpu)
        epochs: 微调轮次
    """
    scaler = torch.cuda.amp.GradScaler()
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=1e-6,          # 初始学习率可增大
    #     momentum=0.9,     # 动量缓冲
    #     weight_decay=1e-5, # 权重衰减
    #     nesterov=True     # Nesterov加速
    # )
    # 使用更稳定的优化器配置
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-6,
        betas=(0.8, 0.99),  # 降低动量影响
        eps=1e-6            # 增大分母保护
    )
    # 必须配合梯度裁剪使用
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
    step=0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Finetune Epoch {epoch+1}/{epochs}')
        
        for image1, image2, flow_gt, valid in progress_bar:
            optimizer.zero_grad()
            img1 = image1.to(device)
            img2 = image2.to(device)
            flow_gt = flow_gt.to(device)
            valid = valid.to(device)
            # img1 = ima1.half()
            # img2 = img2.half()

            model.init_bhwd(img1.shape[0], img1.shape[-2], img1.shape[-1], device)

            with torch.cuda.amp.autocast(enabled=True):
                flow_preds = model(img1, img2, iters_s16=4, iters_s8=7)
                loss, metrics = flow_loss_func(flow_preds, flow_gt, valid, 400)

            scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    # print(f"NaN梯度出现在: {name}")
                    param.grad = torch.nan_to_num(param.grad)
            # print("关键项精度检查:")
            # print(f"梯度 dtype: {next(p.grad.dtype for p in model.parameters() if p.grad is not None)}")  # 
            bad_grad = False
            for name, param in model.named_parameters():
                if not torch.all(torch.isfinite(param.grad)):
                    bad_grad = True
                if bad_grad:
                    pass
                    # print(name, param.grad.mean().item())
                # print(name, torch.max(torch.abs(param.grad)).item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)

            scaler.update()

            # print(epoch,round(metrics['epe'], 3), round(metrics['mag'], 3), optimizer.param_groups[-1]['lr'])
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
       
        
        print(f'Epoch {epoch+1} Average Loss: {epoch_loss/len(dataloader):.4f}')
        step+=1
        if step%20==0:
            torch.save(model, f'./prune/finetune_backbone_{i}_{step}.pth')
            print('model saved to:', f'./prune/finetune_backbone_{i}_{step}.pth')
        log_file = './prune/log_results.txt'
        with open(log_file, 'a') as f:
            f.write(f'Epoch {step} Average Loss: {epoch_loss/len(dataloader):.4f}')
            f.write('\n')

    model.eval()  # 微调完成后切换回评估模式
    return model
    
device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
model = NeuFlow().to(device)
checkpoint = torch.load('./neuflow_things.pth', map_location='cuda', weights_only=True)
model.load_state_dict(checkpoint['model'], strict=True)
# model = torch.load('./prune/pruned_model_backbone_0.pth',weights_only=False) 


# print(model.backbone)

crop_size = (368, 768)
aug_params = {'crop_size': crop_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
kitti2012 = datasets.KITTI(aug_params, split='training',year=2012)
# kitti2015 = datasets.KITTI(aug_params, split='training',year=2015)
aug_params = {'crop_size': crop_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
sintel_clean = datasets.MpiSintelSubset(aug_params, split='training', dstype='clean', subset_ratio=0.3)
sintel_final = datasets.MpiSintelSubset(aug_params, split='training', dstype='final', subset_ratio=0.3)
aug_params = {'crop_size': crop_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
chair = datasets.FlyingChairsSubset(aug_params, split='training')
dataset = sintel_clean + sintel_final + kitti2012 + chair

    
iterative_steps = 1 # You can prune your model to the target pruning ratio iteratively.
i=0
record=False

finetune_loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=16, 
    shuffle=True  # 微调时需要打乱数据
)
log_file = './prune/log_results.txt'
        
# for input_id in range(len(dataset)):
while i < iterative_steps:
    image1, image2, flow_gt, _ = dataset[i]
    image1 = image1.half()
    image2 = image2.half()
    model.half()

    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    
    model.init_bhwd(image1.shape[0], image1.shape[-2], image1.shape[-1], device)

    if not record:
        base_macs, base_nparams = tp.utils.count_ops_and_params(model, (image1,image2))#
        record=True

    # 1. Prune model here
    model =  structured_prune_model(model,image1,image2,device=device)
    # print(model)
    print(" Iter %d, 实际激活参数: %.2f M ; 不考虑稀疏性激活参数: %.2f M ; Base %.2f M"
    % (i+1,count_active_params(model)/1e6,count_active_params_all(model)/1e6,base_nparams / 1e6))
    # with open(log_file, 'a') as f:
    #     f.write('\n\n')
    #     f.write(" Iter %d, 实际激活参数: %.2f M ; 不考虑稀疏性激活参数: %.2f M ; Base %.2f M"
    #     % (i+1,count_active_params(model)/1e6,count_active_params_all(model)/1e6,base_nparams / 1e6))
    #     f.write('\n\n')
    print("="*16)
    # 2. finetune your model here
    print("开始微调")
    model = finetune(model, finetune_loader, device, epochs=160,i=i)
    print("微调完成")
    model.zero_grad() # We don't want to store gradient information
    torch.save(model, f'./prune/pruned_model_backbone_{i}.pth')
    i+=1
print(model.backbone)

# 4. Save & Load
model.zero_grad() # clear gradients to avoid a large file size
torch.save(model, './prune/pruned_model_backbone.pth') # !! no .state_dict here since the structure has been changed after pruning



        
        
