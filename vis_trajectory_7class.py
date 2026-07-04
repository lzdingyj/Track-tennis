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

# ---- visibility categories (4 charts) ----
VIS_CATEGORIES = [
 (0, 'vis0_occluded',       '可见性=0 完全遮挡'),
 (1, 'vis1_visible',        '可见性=1 完全可见'),
 (2, 'vis2_partial',        '可见性=2 部分遮挡'),
 (3, 'vis3_out_of_bounds',  '可见性=3 出界超框'),
]

# ---- status categories (3 charts) ----
STATUS_CATEGORIES = [
 (0, 'stat0_no_track',     '状态=0 无球追踪'),
 (1, 'stat1_in_motion',    '状态=1 球在运动中'),
 (2, 'stat2_bounce_hit',   '状态=2 球触地/击球'),
]


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
     seq_len = seq_len or 8
     bg_mode = bg_mode or ''
     if any(k.startswith('stage1.') for k in ckpt.keys()):
         if model_type == 'TrackNetV3':
             state_dict = ckpt
         else:
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
     model.load_state_dict(state_dict, strict=False)
 else:
     model.load_state_dict(state_dict)
 print(f'  Model: {model_type}, seq_len={seq_len}, bg_mode={bg_mode!r}')
 return model, model_type, seq_len, bg_mode


def draw_trajectory_subset(bg_img, gt_points, pred_points, frame_indices,
                             save_path, title, gt_radius=5, pred_radius=3):
     """Draw GT + predicted points for a subset of frames. No connecting lines.
 
     GT: large filled green circles with frame number.
     Pred: small filled red circles with frame number.
     """
     if isinstance(bg_img, np.ndarray):
         bg_img = Image.fromarray(cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB))
     draw = ImageDraw.Draw(bg_img)
 
     sorted_idx = sorted(frame_indices)
 
     # GT trajectory (green, large dots, no lines)
     for idx in sorted_idx:
         x, y, vis = gt_points[idx]
         if not vis or (x == 0 and y == 0):
             continue
         pt = (int(x), int(y))
         r = gt_radius
         draw.ellipse((pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r),
                      fill='lime', outline='green')
         draw.text((pt[0]+r+2, pt[1]-r), str(idx), fill='green')
 
     # Predicted trajectory (red, smaller dots, no lines)
     for idx in sorted_idx:
         x, y, vis = pred_points[idx]
         if not vis or (x == 0 and y == 0):
             continue
         pt = (int(x), int(y))
         r = pred_radius
         draw.ellipse((pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r),
                      fill='orange', outline='red')
         draw.text((pt[0]+r+2, pt[1]-r+12), str(idx), fill='red')
 
     # Legend
     n = len(sorted_idx)
     draw.rectangle([10, 10, 310, 85], fill=(30, 30, 30))
     draw.text((15, 12), f'{title}  ({n} frames)', fill='white')
     draw.ellipse((15, 32, 31, 48), fill='lime', outline='green')
     draw.text((37, 32), 'Ground Truth', fill='green')
     draw.ellipse((15, 52, 31, 68), fill='orange', outline='red')
     draw.text((37, 52), 'Prediction', fill='red')
 
     bg_img.save(save_path)
     print(f'  Saved: {save_path}')


def main():
 parser = argparse.ArgumentParser(
     description='7 trajectory charts by visibility (4) and status (3).')
 parser.add_argument('--img_dir', type=str, required=True,
                     help='Directory with frame images (0000.jpg, 0001.jpg, ...)')
 parser.add_argument('--gt_csv', type=str, default='',
                     help='GT CSV with columns: file name, visibility, x-coordinate, y-coordinate, status')
 parser.add_argument('--tracknet_file', type=str, required=True,
                     help='TrackNet checkpoint path')
 parser.add_argument('--inpaintnet_file', type=str, default='',
                     help='InpaintNet checkpoint path')
 parser.add_argument('--model_type', type=str, default=None,
                     choices=SUPPORTED_MODELS)
 parser.add_argument('--seq_len', type=int, default=None)
 parser.add_argument('--bg_mode', type=str, default=None,
                     choices=['', 'subtract', 'subtract_concat', 'concat'])
 parser.add_argument('--output_dir', type=str, default='',
                     help='Output directory for the 7 trajectory images')
 parser.add_argument('--batch_size', type=int, default=8)
 parser.add_argument('--eval_mode', type=str, default='weight',
                     choices=['nonoverlap', 'average', 'weight'])
 args = parser.parse_args()

 # ---- 1. Ensure Label.csv ----
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

 # ---- 3. Inference ----
 if args.output_dir:
     os.makedirs(args.output_dir, exist_ok=True)
     save_heatmap_dir = os.path.join(args.output_dir, 'heatmaps')
     os.makedirs(save_heatmap_dir, exist_ok=True)
 else:
     save_heatmap_dir = None

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

 print('Running inference ...')
 with torch.no_grad():
     pred_dict = test_rally((tracknet, inpaintnet), args.img_dir, param_dict,
                            save_heatmap_dir=save_heatmap_dir)

 # ---- 4. Read GT CSV ----
 gt_df = pd.read_csv(label_csv)
 gt_points = []
 for _, row in gt_df.iterrows():
     gt_points.append((
         row.get('x-coordinate', row.get('X', 0)),
         row.get('y-coordinate', row.get('Y', 0)),
         row.get('visibility', row.get('Visibility', 0)),
     ))
 n_total = len(gt_points)

 # ---- 5. Predicted points ----
 pred_points = []
 for i in range(len(pred_dict['Frame'])):
     pred_points.append((
         pred_dict['X'][i],
         pred_dict['Y'][i],
         pred_dict['Visibility'][i],
     ))
 while len(pred_points) < n_total:
     pred_points.append((0, 0, 0))
 pred_points = pred_points[:n_total]

 # ---- 6. Read visibility & status arrays ----
 vis_col = gt_df['visibility'].values if 'visibility' in gt_df.columns else gt_df['Visibility'].values
 has_status = 'status' in gt_df.columns
 if has_status:
     raw = gt_df['status'].fillna(-1).values
     status_col = np.array([-1 if str(s).strip() == '' else int(s) for s in raw])

 # ---- 7. Background image ----
 last_idx = n_total - 1
 bg_path = os.path.join(args.img_dir, f'{last_idx:04d}.{IMG_FORMAT}')
 if not os.path.exists(bg_path):
     bg_path = os.path.join(args.img_dir, f'0000.{IMG_FORMAT}')

 output_dir = args.output_dir or '.'
 os.makedirs(output_dir, exist_ok=True)

 # ---- 8. 4 visibility-based charts ----
 print('\n--- Visibility-based charts ---')
 for vis_val, tag, label in VIS_CATEGORIES:
     indices = [i for i in range(n_total) if vis_col[i] == vis_val]
     if len(indices) < 2:
         print(f'  [{tag}] {label}: {len(indices)} frames (skip, <2)')
         continue
     bg = Image.open(bg_path).convert('RGB')
     draw_trajectory_subset(bg, gt_points, pred_points, indices,
                            os.path.join(output_dir, f'{tag}.png'), label)

 # ---- 9. 3 status-based charts (if status column exists) ----
 if has_status:
     print('\n--- Status-based charts ---')
     for stat_val, tag, label in STATUS_CATEGORIES:
         indices = [i for i in range(n_total) if status_col[i] == stat_val]
         if len(indices) < 2:
             print(f'  [{tag}] {label}: {len(indices)} frames (skip, <2)')
             continue
         bg = Image.open(bg_path).convert('RGB')
         draw_trajectory_subset(bg, gt_points, pred_points, indices,
                                os.path.join(output_dir, f'{tag}.png'), label)
 else:
     print('\n  (No "status" column in CSV, skipped status-based charts.)')

 print(f'\nDone. {len(os.listdir(output_dir))} charts in: {output_dir}')


if __name__ == '__main__':
 main()
