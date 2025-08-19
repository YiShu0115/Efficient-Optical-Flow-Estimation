import torch
from torch.utils.data import DataLoader

import argparse
import os

from data_utils.datasets import build_train_dataset
from NeuFlow.neuflow import NeuFlow,NeuFlowOri
from loss import flow_loss_func
from data_utils.evaluate import validate_things, validate_sintel, validate_kitti, validate_viper
from load_model import my_load_weights, my_freeze_model
from dist_utils import get_dist_info, init_dist, setup_for_distributed
from NeuFlow.loss import SequenceLoss
# 在Python代码中添加

print(f"Allocated: {torch.cuda.memory_allocated()/1e9} GB")
print(f"Cached: {torch.cuda.memory_reserved()/1e9} GB")

def save_val_epe_to_txt(step, val_epe_sintel, val_epe_kitti, save_dir):
    save_path = os.path.join(save_dir, "val_epe.txt")
    with open(save_path, 'a') as f:
        f.write(f"{step} {val_epe_sintel:.6f} {val_epe_kitti:.6f}\n")

def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default=None, type=str)
    parser.add_argument('--dataset_dir', default=None, type=str)
    parser.add_argument('--stage', default='things', type=str)
    parser.add_argument('--val_dataset', default=['things', 'sintel'], type=str, nargs='+')

    # training
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--val_freq', default=1000, type=int)
    parser.add_argument('--num_steps', default=1000000, type=int)
    parser.add_argument('--start_step', default=0, type=int)

    parser.add_argument('--max_flow', default=400, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--strict_resume', action='store_true')

    # distributed training
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')

    return parser

def main(args):
    print('Use %d GPUs' % torch.cuda.device_count())
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()  # 释放未使用的显存
    if args.distributed:
        # assert args.batch_size % torch.cuda.device_count() == 0
        # args.batch_size = args.batch_size // torch.cuda.device_count()
        # dist_params = dict(backend='nccl')
        # init_dist('pytorch', **dist_params)
        # _, world_size = get_dist_info()
        # args.gpu_ids = range(world_size)
        # device = torch.device('cuda:{}'.format(args.local_rank))
        # setup_for_distributed(args.local_rank == 0)
            # 确保从环境变量获取正确的local_rank
        args.local_rank = int(os.environ['LOCAL_RANK'])
        
        # 关键修改1：先设置设备再处理batch_size
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f'cuda:{args.local_rank}')
        
        # 关键修改2：从环境变量获取world_size而不是通过device_count
        _, world_size = get_dist_info()
        assert args.batch_size % world_size == 0, \
            f"Batch size {args.batch_size} must be divisible by world size {world_size}"
        args.batch_size = args.batch_size // world_size
        dist_params = dict(backend='nccl')
        init_dist('pytorch', **dist_params)
        # 初始化分布式
        # dist.init_process_group(
        #     backend='nccl',
        #     init_method='env://',
        #     world_size=world_size,
        #     rank=int(os.environ['RANK']))
        args.gpu_ids = range(world_size)
        setup_for_distributed(args.local_rank == 0)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = NeuFlow().to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank)
        model_without_ddp = model.module
        # model = torch.nn.parallel.DistributedDataParallel(
        #     model,
        #     device_ids=[args.local_rank],
        #     output_device=args.local_rank,
        #     find_unused_parameters=True  # 如果模型有分支可能需要这个
        # )
        # model_without_ddp = model.module
    else:
        model_without_ddp = model

    num_params = sum(p.numel() for p in model.parameters())
    print('Number of params:', num_params)

    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                  weight_decay=1e-4)

    start_step = args.start_step
    save_dir = f"./training_plots_v2"
    os.makedirs(save_dir, exist_ok=True)

    if args.resume:
        state_dict = my_load_weights(args.resume)
        model_without_ddp.load_state_dict(state_dict, strict=args.strict_resume)
        my_freeze_model(model)
        torch.save({
            'model': model_without_ddp.state_dict()
        }, os.path.join(args.checkpoint_dir, 'step_0.pth'))

    train_dataset = build_train_dataset(args.stage)
    print('Number of training images:', len(train_dataset))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=torch.cuda.device_count(),
            rank=args.local_rank)
    else:
        train_sampler = None

    shuffle = False if args.distributed else True
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=shuffle, num_workers=args.num_workers,
                                               pin_memory=True, drop_last=True,
                                               sampler=train_sampler)

    total_steps = start_step
    epoch = 0
    counter = 0
    teacher = NeuFlowOri().to(device)
    teacher_checkpoint = torch.load("neuflow_sintel.pth", map_location='cuda', weights_only=True)
    teacher.load_state_dict(teacher_checkpoint['model'], strict=True)
    teacher.eval()

    val_epe_sintel_history = []
    val_epe_kitti_history = []

    while total_steps < args.num_steps:
        model.train()
        if args.distributed:
            train_sampler.set_epoch(epoch)

        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()
            img1, img2, flow_gt, valid = [x.to(device) for x in sample]
            img1 = img1.to(dtype=torch.float16)
            img2 = img2.to(dtype=torch.float16)
            model_without_ddp.init_bhwd(img1.shape[0], img1.shape[-2], img1.shape[-1], device)

            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(img1, img2, iters_s16=4, iters_s8=7)
                loss_fn = SequenceLoss(gamma=0.9, max_flow=400, var_min=0, var_max=10)
                loss, metrics = loss_fn(outputs, flow_gt, valid, args.max_flow)
                with torch.no_grad():
                    teacher.init_bhwd(img1.shape[0], img2.shape[-2], img1.shape[-1], device)
                    teacher_outputs = teacher(img1, img2)
                    teacher_flow = teacher_outputs[-1]
                L2_loss = torch.mean((outputs['flow_preds'][-1] - teacher_flow) ** 2)
                alpha = 0.1
                total_loss = (1 - alpha) * loss + alpha * L2_loss

            scaler.scale(total_loss).backward(retain_graph=True)
            scaler.unscale_(optimizer)

            bad_grad = False
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"警告: {name} 的梯度为None (可能未使用)")
                    continue
                if not torch.all(torch.isfinite(param.grad)):
                    bad_grad = True
                if bad_grad:
                    print(name, param.grad.mean().item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            print(total_steps, round(metrics['epe'], 3), round(metrics['mag'], 3), optimizer.param_groups[-1]['lr'])
            total_steps += 1

            if total_steps % args.val_freq == 0:
                if args.local_rank == 0:
                    checkpoint_path = os.path.join(args.checkpoint_dir, 'step_%06d.pth' % total_steps)
                    torch.save({
                        'model': model_without_ddp.state_dict()
                    }, checkpoint_path)

                val_results = {}
                val_epe_sintel = 0.0
                val_epe_kitti = 0.0
                if 'things' in args.val_dataset:
                    test_results_dict = validate_things(model_without_ddp, device, dstype='frames_cleanpass', validate_subset=True)
                    if args.local_rank == 0:
                        val_results.update(test_results_dict)
                if 'sintel' in args.val_dataset:
                    test_results_dict = validate_sintel(model_without_ddp, device, dstype='final')
                    if args.local_rank == 0:
                        val_results.update(test_results_dict)
                        val_epe_sintel = test_results_dict.get('sintel_final_epe', 0.0)
                        val_epe_sintel_history.append(val_epe_sintel)
                if 'kitti' in args.val_dataset:
                    test_results_dict = validate_kitti(model_without_ddp, device)
                    if args.local_rank == 0:
                        val_results.update(test_results_dict)
                        val_epe_kitti = test_results_dict.get('kitti_epe', 0.0)
                        val_epe_kitti_history.append(val_epe_kitti)
                if 'viper' in args.val_dataset:
                    test_results_dict = validate_viper(model_without_ddp, device)
                    if args.local_rank == 0:
                        val_results.update(test_results_dict)

                # Save validation EPE to txt after validation
                if args.local_rank == 0:
                    save_val_epe_to_txt(total_steps, val_epe_sintel, val_epe_kitti, save_dir)

                if args.local_rank == 0:
                    counter += 1
                    if counter >= 10:
                        for group in optimizer.param_groups:
                            group['lr'] *= 0.8
                        counter = 0
                    val_file = os.path.join(args.checkpoint_dir, 'val_results.txt')
                    with open(val_file, 'a') as f:
                        f.write('step: %06d lr: %.6f\n' % (total_steps, optimizer.param_groups[-1]['lr']))
                        for k, v in val_results.items():
                            f.write("| %s: %.3f " % (k, v))
                        f.write('\n\n')

                model.train()
        epoch += 1

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    
    # 关键：确保每个进程使用不同的GPU

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    # local_rank = int(os.environ['LOCAL_RANK'])
    # torch.cuda.set_device(local_rank)
    main(args)