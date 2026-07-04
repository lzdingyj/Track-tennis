import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Shuttlecock_Trajectory_Dataset
from test import eval_tracknet, eval_inpaintnet
from utils.general import ResumeArgumentParser, to_img_format
from utils.metric import WBCELoss
from utils.visualize import plot_heatmap_pred_sample, plot_traj_pred_sample

try:
    from model import TrackNet_Baseline, TrackNet_Ghost, TrackNet_GhostAtt
except:
    import sys

    sys.path.append('.')
    from model import TrackNet_Baseline, TrackNet_Ghost, TrackNet_GhostAtt


def mixup(x, y, alpha=0.5):
    batch_size = x.size()[0]
    lamb = np.random.beta(alpha, alpha, size=batch_size)
    lamb = np.maximum(lamb, 1 - lamb)
    lamb = torch.from_numpy(lamb[:, None, None, None]).float().to(x.device)
    index = torch.randperm(batch_size)
    x_mix = x * lamb + x[index] * (1 - lamb)
    y_mix = y * lamb + y[index] * (1 - lamb)
    return x_mix, y_mix


def get_random_mask(mask_size, mask_ratio):
    mask = np.random.binomial(1, mask_ratio, size=mask_size)
    mask = torch.from_numpy(mask).float().cuda().unsqueeze(-1)
    return mask


def train_tracknet(model, optimizer, data_loader, param_dict):
    model.train()
    epoch_loss = []
    if param_dict['verbose']:
        data_prob = tqdm(data_loader)
    else:
        data_prob = data_loader

    for step, (_, x, y, c, _) in enumerate(data_prob):
        optimizer.zero_grad()
        x, y = x.float().cuda(), y.float().cuda()
        if param_dict['alpha'] > 0:
            x, y = mixup(x, y, param_dict['alpha'])

        y_pred = model(x)
        loss = WBCELoss(y_pred, y)
        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    return float(np.mean(epoch_loss))


def train_inpaintnet(model, optimizer, data_loader, param_dict):
    model.train()
    epoch_loss = []
    if param_dict['verbose']:
        data_prob = tqdm(data_loader)
    else:
        data_prob = data_loader

    for step, (_, coor_pred, coor_gt, _, vis_gt, _) in enumerate(data_prob):
        optimizer.zero_grad()
        coor_pred, coor_gt, vis_gt = coor_pred.float().cuda(), coor_gt.float().cuda(), vis_gt.float().cuda()
        mask = get_random_mask(mask_size=coor_gt.shape[:2], mask_ratio=param_dict['mask_ratio']).cuda()
        inpaint_mask = torch.logical_and(vis_gt, mask).int()
        coor_pred = coor_pred * (1 - inpaint_mask)
        refine_coor = model(coor_pred, inpaint_mask)
        masked_refine_coor = refine_coor * inpaint_mask
        masked_gt_coor = coor_gt * inpaint_mask
        loss = nn.MSELoss()(masked_refine_coor, masked_gt_coor)
        epoch_loss.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    return float(np.mean(epoch_loss))


def run_training(model_name, save_dir, total_epochs=20):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=model_name)
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=total_epochs)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_scheduler', type=str, default='')
    parser.add_argument('--bg_mode', type=str, default='')
    parser.add_argument('--alpha', type=float, default=-1)
    parser.add_argument('--frame_alpha', type=float, default=-1)
    parser.add_argument('--mask_ratio', type=float, default=0.3)
    parser.add_argument('--tolerance', type=float, default=4)
    parser.add_argument('--resume_training', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--save_dir', type=str, default=save_dir)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=True)

    args = parser.parse_args([])
    param_dict = vars(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print(f"\n===== 训练: {model_name} | 保存到: {save_dir} | 轮数: {total_epochs} =====")

    global display_step
    display_step = 4 if args.debug else 100
    num_workers = args.batch_size if args.batch_size <= 16 else 16

    train_dataset = Shuttlecock_Trajectory_Dataset(
        split='train', seq_len=args.seq_len, sliding_step=1,
        data_mode='heatmap', bg_mode=args.bg_mode, frame_alpha=args.frame_alpha, debug=args.debug
    )
    val_dataset = Shuttlecock_Trajectory_Dataset(
        split='val', seq_len=args.seq_len, sliding_step=args.seq_len,
        data_mode='heatmap', bg_mode=args.bg_mode, debug=args.debug
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers,
                            drop_last=False, pin_memory=True)

    print(f'创建模型: {args.model_name}...')

    # ===================== 【修复】输出通道统一为 8 =====================
    if model_name == "TrackNet_Baseline":
        model = TrackNet_Baseline(in_dim=24, out_dim=8).cuda()
    elif model_name == "TrackNet_Ghost":
        model = TrackNet_Ghost(in_dim=24, out_dim=8).cuda()
    elif model_name == "TrackNet_GhostAtt":
        model = TrackNet_GhostAtt(in_dim=24, out_dim=8, num_frames=8).cuda()
    else:
        raise ValueError("模型不存在")

    train_fn = train_tracknet
    eval_fn = eval_tracknet

    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.learning_rate)

    scheduler = None
    max_val_acc = 0.0

    print(f'Start training...')
    train_start_time = time.time()
    for epoch in range(args.epochs):
        print(f'Epoch [{epoch + 1} / {args.epochs}]')
        start_time = time.time()
        train_loss = train_fn(model, optimizer, train_loader, param_dict)
        val_loss, val_res = eval_fn(model, val_loader, param_dict)

        cur_val_acc = val_res['accuracy']
        if cur_val_acc >= max_val_acc:
            max_val_acc = cur_val_acc
            torch.save({
                'epoch': epoch, 'max_val_acc': max_val_acc,
                'model': model.state_dict(), 'optimizer': optimizer.state_dict()
            }, os.path.join(args.save_dir, f'{model_name}_best.pt'))

        torch.save({
            'epoch': epoch, 'max_val_acc': max_val_acc,
            'model': model.state_dict(), 'optimizer': optimizer.state_dict()
        }, os.path.join(args.save_dir, f'{model_name}_cur.pt'))

        print(f'耗时: {(time.time() - start_time) / 3600:.2f}h | Val Acc: {cur_val_acc:.4f}')

    print(f'{model_name} 训练完成！总时间: {(time.time() - train_start_time) / 3600:.2f}h\n')


if __name__ == '__main__':
    run_training("TrackNet_Baseline", "exp_baseline", 20)
    run_training("TrackNet_Ghost", "exp_ghost", 20)
    run_training("TrackNet_GhostAtt", "exp_ghost_att", 20)
    print("===== 所有消融实验全部完成 ✅ =====")