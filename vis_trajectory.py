import os
import shutil
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict
from PIL import Image, ImageDraw

import torch
from torch.utils.data import DataLoader

from dataset import Shuttlecock_Trajectory_Dataset, data_dir
from utils.general import *
from test import test_rally, evaluate, get_ensemble_weight, generate_inpaint_mask

SUPPORTED_MODELS = ['TrackNetV3Improved', 'TrackNet_Baseline',
                    'TrackNet_Ghost', 'TrackNet_GhostAtt', 'TrackNetV3']


def load_model_from_ckpt(ckpt_path, model_type=None, seq_len=None, bg_mode=None):
    ckpt = torch.load(ckpt_path, map_location='cuda')

    has_param_dict = 'param_dict' in ckpt

    if has_param_dict and model_type is None:
        pd_ = ckpt['param_dict']
        model_type = pd_.get('model_name', 'TrackNet')
        seq_len    = pd_.get('seq_len', 8)
        bg_mode    = pd_.get('bg_mode', '')
        state_dict = ckpt['model']
    elif has_param_dict:
        pd_ = ckpt['param_dict']
        state_dict = ckpt['model']
        seq_len    = seq_len    or pd_.get('seq_len', 8)
        bg_mode    = bg_mode    or pd_.get('bg_mode', '')
    else:
        # no param_dict in checkpoint
        seq_len = seq_len or 8
        bg_mode = bg_mode or ''
        # Checkpoint with 'stage1.' prefix keys
        if any(k.startswith('stage1.') for k in ckpt.keys()):
            if model_type == 'TrackNetV3':
                # TrackNetV3 model expects stage1.xxx → use as-is
                state_dict = ckpt
            else:
                # TrackNet_Baseline etc. → strip stage1. prefix
                state_dict = OrderedDict(
                    (k[len('stage1.'):], v)
                    for k, v in ckpt.items()
                    if k.startswith('stage1.')
                )
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt

    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f'Unsupported model: {model_type}. Choose from: {SUPPORTED_MODELS}')

    model = get_model(model_type, seq_len=seq_len, bg_mode=bg_mode).cuda()
    if model_type == 'TrackNetV3':
        # TrackNetV3 has stage1 + stage2; checkpoint may only have stage1 weights
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict)
    print(f'  Model: {model_type}, seq_len={seq_len}, bg_mode={bg_mode!r}')
    return model, model_type, seq_len, bg_mode


def draw_trajectory_image(bg_img, gt_points, pred_points, save_path, radius=5):
    if isinstance(bg_img, np.ndarray):
        bg_img = Image.fromarray(cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(bg_img)

    def _draw_traj(points, outline_color, fill_color):
        prev = None
        for i, (x, y, vis) in enumerate(points):
            if not vis or (x == 0 and y == 0):
                prev = None
                continue
            pt = (int(x), int(y))
            if prev is not None:
                draw.line([prev, pt], fill=outline_color, width=2)
            draw.ellipse((pt[0]-radius, pt[1]-radius, pt[0]+radius, pt[1]+radius),
                         fill=fill_color, outline=outline_color)
            draw.text((pt[0]+radius+2, pt[1]-radius), str(i), fill=outline_color)
            prev = pt

    _draw_traj(gt_points, outline_color='green', fill_color='lime')
    _draw_traj(pred_points, outline_color='red', fill_color='orange')

    draw.rectangle([10, 10, 180, 60], fill=(30, 30, 30))
    draw.ellipse((20, 20, 36, 36), fill='lime', outline='green')
    draw.text((42, 20), 'Ground Truth', fill='green')
    draw.ellipse((20, 40, 36, 56), fill='orange', outline='red')
    draw.text((42, 40), 'Prediction', fill='red')

    bg_img.save(save_path)
    print(f'Trajectory image saved to {save_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Visualise GT + predicted trajectories on a single image.')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Directory with frame images (0000.jpg, 0001.jpg, ...)')
    parser.add_argument('--gt_csv', type=str, default='',
                        help='GT CSV (columns: visibility,x-coordinate,y-coordinate)')
    parser.add_argument('--tracknet_file', type=str, required=True,
                        help='TrackNet checkpoint path')
    parser.add_argument('--inpaintnet_file', type=str, default='',
                        help='InpaintNet checkpoint path')
    parser.add_argument('--model_type', type=str, default=None,
                        choices=SUPPORTED_MODELS,
                        help='Model architecture (auto-detected from checkpoint if omitted)')
    parser.add_argument('--seq_len', type=int, default=None,
                        help='TrackNet seq_len (auto-detected from checkpoint if omitted)')
    parser.add_argument('--bg_mode', type=str, default=None,
                        choices=['', 'subtract', 'subtract_concat', 'concat'],
                        help='Background mode (auto-detected from checkpoint if omitted)')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Output directory for trajectory image + heatmaps (overrides --save_path)')
    parser.add_argument('--save_path', type=str, default='trajectory_output.png')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_mode', type=str, default='weight',
                        choices=['nonoverlap', 'average', 'weight'])
    args = parser.parse_args()

    # ---- 1. Ensure Label.csv exists ----
    label_csv = os.path.join(args.img_dir, 'Label.csv')
    if args.gt_csv:
        shutil.copy2(args.gt_csv, label_csv)
        print(f'Copied GT CSV -> {label_csv}')
    elif not os.path.exists(label_csv):
        print(f'ERROR: No Label.csv found in {args.img_dir} and no --gt_csv provided.')
        return

    # ---- 2. Load model ----
    print('Loading TrackNet...')
    tracknet, model_type, seq_len, bg_mode = load_model_from_ckpt(
        args.tracknet_file, args.model_type, args.seq_len, args.bg_mode)

    inpaintnet = None
    if args.inpaintnet_file:
        print('Loading InpaintNet...')
        inp_ckpt = torch.load(args.inpaintnet_file, map_location='cuda')
        inpaintnet = get_model('InpaintNet').cuda()
        inpaintnet.load_state_dict(inp_ckpt['model'])

    # ---- 3. param_dict and inference ----
    param_dict = {
        'tracknet_model_name': model_type,
        'tracknet_seq_len': seq_len,
        'bg_mode': bg_mode,
        'eval_mode': args.eval_mode,
        'tolerance': 4,
        'batch_size': args.batch_size,
        'num_workers': 0,
        'output_bbox': False,
        'output_gt': True,
        'verbose': True,
        'debug': False,
    }

    # Handle output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        save_heatmap_dir = os.path.join(args.output_dir, 'heatmaps')
        os.makedirs(save_heatmap_dir, exist_ok=True)
        trajectory_path = os.path.join(args.output_dir, os.path.basename(args.save_path))
    else:
        save_heatmap_dir = None
        trajectory_path = args.save_path

    print('Running inference ...')
    with torch.no_grad():
        pred_dict = test_rally((tracknet, inpaintnet), args.img_dir, param_dict,
                               save_heatmap_dir=save_heatmap_dir)

    # ---- 4. Read GT points from CSV ----
    gt_df = pd.read_csv(label_csv)
    gt_points = []
    for _, row in gt_df.iterrows():
        gt_points.append((
            row.get('x-coordinate', row.get('X', 0)),
            row.get('y-coordinate', row.get('Y', 0)),
            row.get('visibility', row.get('Visibility', 0)),
        ))

    # ---- 5. Predicted points ----
    pred_points = []
    for i in range(len(pred_dict['Frame'])):
        pred_points.append((
            pred_dict['X'][i],
            pred_dict['Y'][i],
            pred_dict['Visibility'][i],
        ))
    while len(pred_points) < len(gt_points):
        pred_points.append((0, 0, 0))
    pred_points = pred_points[:len(gt_points)]

    # ---- 6. Load background (last frame) ----
    last_idx = len(gt_points) - 1
    last_img = os.path.join(args.img_dir, f'{last_idx:04d}.{IMG_FORMAT}')
    if not os.path.exists(last_img):
        last_img = os.path.join(args.img_dir, f'0000.{IMG_FORMAT}')
    bg = Image.open(last_img).convert('RGB')

    # ---- 7. Draw ----
    draw_trajectory_image(bg, gt_points, pred_points, trajectory_path, radius=5)


if __name__ == '__main__':
    main()
