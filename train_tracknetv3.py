import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import Shuttlecock_Trajectory_Dataset
from utils.metric import WBCELoss
from model import TrackNetV3


def mixup(x, y, alpha=0.5):
    batch_size = x.size()[0]
    lamb = np.random.beta(alpha, alpha, size=batch_size)
    lamb = np.maximum(lamb, 1 - lamb)
    lamb = torch.from_numpy(lamb[:, None, None, None]).float().to(x.device)
    index = torch.randperm(batch_size)
    x_mix = x * lamb + x[index] * (1 - lamb)
    y_mix = y * lamb + y[index] * (1 - lamb)
    return x_mix, y_mix


# =============================================================================
# ✅ Stage2 训练：轨迹修复（InpaintNet）
# =============================================================================
def train_stage2(model, optimizer, loader, args):
    model.stage1.eval()
    for p in model.stage1.parameters():
        p.requires_grad = False
    model.stage2.train()

    total_loss = 0
    for batch in tqdm(loader, desc="Stage2 Training"):
        optimizer.zero_grad()
        _, x, _, y_coord, _ = batch[:5]
        x = x.cuda().float()
        y_coord = y_coord.cuda().float()

        _, pred_coord = model(x, return_coords=True)
        loss = F.mse_loss(pred_coord, y_coord)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def val_stage2(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            _, x, _, y_coord, _ = batch[:5]
            x = x.cuda().float()
            y_coord = y_coord.cuda().float()
            _, pred_coord = model(x, return_coords=True)
            total_loss += F.mse_loss(pred_coord, y_coord).item()
    return total_loss / len(loader)


# =============================================================================
# 🚀 主函数：完全修复版
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=-1)
    parser.add_argument('--save_dir', default='exp_tracknetv3')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda'

    baseline_ckpt_path = "/home/featurize/work/TrackNetV3_codex-improve/exp_baseline/TrackNet_Baseline_best.pt"

    train_loader = DataLoader(
        Shuttlecock_Trajectory_Dataset(split='train', seq_len=args.seq_len, data_mode='heatmap'),
        batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    val_loader = DataLoader(
        Shuttlecock_Trajectory_Dataset(split='val', seq_len=args.seq_len, data_mode='heatmap'),
        batch_size=args.batch_size, shuffle=False, num_workers=8
    )

    # ✅ 固定 in_dim=24（你训练baseline时用的维度）
    model = TrackNetV3(in_dim=24, out_dim=args.seq_len, num_frames=args.seq_len).to(device)

    # ==========================
    # ✅✅✅ 这里终于 100% 正确 ✅✅✅
    # ==========================
    print(f"✅ 加载预训练 Baseline：{baseline_ckpt_path}")
    checkpoint = torch.load(baseline_ckpt_path, map_location=device)
    model.stage1.load_state_dict(checkpoint["model"])

    # ==========================
    # 训练 Stage2 20轮
    # ==========================
    print("\n===== 开始训练 STAGE 2 (20 EPOCHS) =====")
    opt2 = torch.optim.Adam(model.stage2.parameters(), lr=args.lr)
    best_loss2 = 1e9

    for epoch in range(20):
        t0 = time.time()
        train_loss = train_stage2(model, opt2, train_loader, args)
        val_loss = val_stage2(model, val_loader)

        if val_loss < best_loss2:
            best_loss2 = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'tracknetv3_full_best.pth'))

        print(f"Stage2 Epoch {epoch+1:2d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | {time.time()-t0:.1f}s")

    print("\n✅ 训练完成！最终完整模型：exp_tracknetv3/tracknetv3_full_best.pth")