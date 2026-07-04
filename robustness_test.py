import os, sys, shutil, json, copy, argparse
from collections import OrderedDict
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import Shuttlecock_Trajectory_Dataset, data_dir
from utils.general import *
from test import test_rally, pred_types, pred_types_map, get_metric

# ====================  Perturbation Functions ====================

def apply_brightness(src_path, dst_path, factor):
    """Brightness adjustment: factor < 1 = darker, > 1 = brighter."""
    img = cv2.imread(src_path)
    img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
    cv2.imwrite(dst_path, img)


def apply_noise(src_path, dst_path, std):
    """Gaussian noise with given standard deviation."""
    img = cv2.imread(src_path).astype(np.float32)
    noise = np.random.randn(*img.shape) * std
    result = np.clip(img + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(dst_path, result)


def apply_contrast(src_path, dst_path, factor):
    """Contrast scaling: factor < 1 reduces, > 1 increases contrast."""
    img = cv2.imread(src_path).astype(np.float32)
    img = (img - 128.0) * factor + 128.0
    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    cv2.imwrite(dst_path, img)


def apply_compression(src_path, dst_path, quality):
    """JPEG compression: quality 1-100 (higher = better)."""
    img = cv2.imread(src_path)
    cv2.imwrite(dst_path, img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])


PERTURBATIONS = OrderedDict([
    ("brightness",  {"levels": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0],
                      "desc": "Brightness factor", "func": apply_brightness}),
    ("contrast",    {"levels": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0],
                      "desc": "Contrast factor",   "func": apply_contrast}),
    ("noise",       {"levels": [1, 2, 5, 8, 10, 15, 20, 30, 50, 75, 100],
                      "desc": "Gaussian noise (\u03c3)", "func": apply_noise}),
    ("compression", {"levels": [95, 90, 85, 80, 70, 60, 50, 40, 30, 20, 10, 5],
                      "desc": "JPEG quality",       "func": apply_compression}),
])


# ====================  Metric helpers ====================

def compute_metrics(pred_dict):
    """Compute evaluation metrics from a single-rally pred_dict."""
    types = np.array(pred_dict["Type"])
    counts = {t: int((types == pred_types_map[t]).sum()) for t in pred_types}
    TP, TN, FP1, FP2, FN = counts["TP"], counts["TN"], counts["FP1"], counts["FP2"], counts["FN"]
    acc, prec, rec, f1, miss = get_metric(TP, TN, FP1, FP2, FN)
    return {
        'TP': TP, 'TN': TN, 'FP1': FP1, 'FP2': FP2, 'FN': FN,
        'accuracy': float(f'{acc:.4f}'), 'precision': float(f'{prec:.4f}'),
        'recall': float(f'{rec:.4f}'), 'f1': float(f'{f1:.4f}'),
        'miss_rate': float(f'{miss:.4f}')
    }


def prepare_perturbed_dir(src_dir, dst_dir, pert_name, pert_level, img_format='jpg'):
    """Copy frames with perturbation applied, plus Label.csv."""
    os.makedirs(dst_dir, exist_ok=True)
    apply_func = PERTURBATIONS[pert_name]["func"]

    frames = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    for fname in frames:
        apply_func(os.path.join(src_dir, fname), os.path.join(dst_dir, fname), pert_level)

    label_src = os.path.join(src_dir, 'Label.csv')
    if os.path.exists(label_src):
        shutil.copy2(label_src, os.path.join(dst_dir, 'Label.csv'))


# ====================  Model loading ====================

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
        seq_len = seq_len or 8
        bg_mode = bg_mode or ''
        if any(k.startswith('stage1.') for k in ckpt.keys()):
            if model_type == 'TrackNetV3':
                state_dict = ckpt
            else:
                state_dict = OrderedDict(
                    (k[len('stage1.'):], v) for k, v in ckpt.items() if k.startswith('stage1.')
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
    return model, model_type, seq_len, bg_mode


# ====================  Main ====================

def print_table(results):
    """Pretty-print comparison table."""
    sep = '-' * 130
    header = (f'{"Condition":<30} {"TP":>5} {"TN":>5} {"FP1":>5} {"FP2":>5} {"FN":>5} '
              f'{"Acc":>8} {"Prec":>8} {"Recall":>8} {"F1":>8} {"MissRate":>8}')
    print(sep)
    print(header)
    print(sep)
    for cond, r in results.items():
        print(f'{cond:<30} {r["TP"]:>5} {r["TN"]:>5} {r["FP1"]:>5} {r["FP2"]:>5} {r["FN"]:>5} '
              f'{r["accuracy"]:>8.4f} {r["precision"]:>8.4f} {r["recall"]:>8.4f} '
              f'{r["f1"]:>8.4f} {r["miss_rate"]:>8.4f}')
    print(sep)


def main():
    parser = argparse.ArgumentParser(description='Robustness test: evaluate model under image perturbations.')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory with original frames')
    parser.add_argument('--tracknet_file', type=str, required=True, help='TrackNet checkpoint path')
    parser.add_argument('--gt_csv', type=str, default='', help='GT CSV (copied to Label.csv)')
    parser.add_argument('--model_type', type=str, default=None, choices=SUPPORTED_MODELS)
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--bg_mode', type=str, default=None, choices=['', 'subtract', 'subtract_concat', 'concat'])
    parser.add_argument('--save_dir', type=str, default='robustness_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_mode', type=str, default='weight', choices=['nonoverlap', 'average', 'weight'])
    parser.add_argument('--perturbations', type=str, default='all',
                        help='Comma-separated list: brightness,noise,compression, or "all"')
    
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Clean up any stale perturbation dirs from previous runs
    parent = os.path.dirname(args.img_dir)
    for d in os.listdir(parent):
        if d.startswith('_pert_'):
            shutil.rmtree(os.path.join(parent, d), ignore_errors=True)

    # Also delete stale img_config cache so it gets rebuilt with correct rally count
    from dataset import data_dir
    stale_cfg = os.path.join(data_dir, f'img_config_288x512_test.npz')
    if os.path.exists(stale_cfg):
        os.remove(stale_cfg)
        print(f'Removed stale cache: {stale_cfg}')


    # ---- 1. Ensure Label.csv ----
    label_csv = os.path.join(args.img_dir, 'Label.csv')
    if args.gt_csv:
        shutil.copy2(args.gt_csv, label_csv)
        print(f'Copied GT CSV -> {label_csv}')
    elif not os.path.exists(label_csv):
        print(f'ERROR: No Label.csv in {args.img_dir} and no --gt_csv.')
        return

    # ---- 2. Load model ----
    print('\nLoading TrackNet...')
    tracknet, model_type, seq_len, bg_mode = load_model_from_ckpt(
        args.tracknet_file, args.model_type, args.seq_len, args.bg_mode)
    model_tuple = (tracknet, None)

    param_dict = {
        'tracknet_model_name': model_type,
        'tracknet_seq_len': seq_len,
        'bg_mode': bg_mode,
        'eval_mode': args.eval_mode,
        'tolerance': 4,
        'batch_size': args.batch_size,
        'num_workers': 0,
        'output_bbox': False,
        'output_gt': False,
        'verbose': False,
        'debug': False,
    }

    # ---- 3. Determine which perturbations to run ----
    if args.perturbations == 'all':
        selected = list(PERTURBATIONS.keys())
    else:
        selected = [p.strip() for p in args.perturbations.split(',')]

    # ---- 4. Baseline ----
    print('\n>>> Baseline (original images) ...')
    with torch.no_grad():
        pred_dict = test_rally(model_tuple, args.img_dir, param_dict)
    results = {'baseline': compute_metrics(pred_dict)}

    # ---- 5. Perturbation loop ----
    for pert_name in selected:
        if pert_name not in PERTURBATIONS:
            print(f'  Skipping unknown perturbation: {pert_name}')
            continue
        cfg = PERTURBATIONS[pert_name]
        for level in cfg['levels']:
            cond_name = f'{pert_name}_{level}'
            pert_dir = os.path.join(os.path.dirname(args.img_dir), f'_pert_{cond_name}')
            print(f'\n>>> {cond_name} ({cfg["desc"]}={level}) ...')

            prepare_perturbed_dir(args.img_dir, pert_dir, pert_name, level, 'jpg')
            # Rebuild img_config cache so it includes the new _pert_ dir
            # data_dir already imported at top
            _cfg = os.path.join(data_dir, f'img_config_288x512_test.npz')
            if os.path.exists(_cfg):
                os.remove(_cfg)
            try:
                with torch.no_grad():
                    pred_dict = test_rally(model_tuple, pert_dir, param_dict)
                results[cond_name] = compute_metrics(pred_dict)
                m = results[cond_name]
                print(f'    Acc={m["accuracy"]:.4f}  Prec={m["precision"]:.4f}  '
                      f'Recall={m["recall"]:.4f}  F1={m["f1"]:.4f}  MissRate={m["miss_rate"]:.4f}')
            finally:
                shutil.rmtree(pert_dir, ignore_errors=True)

    # ---- 6. Summary ----
    print('\n\n=============== Robustness Test Results ===============')
    print_table(results)

    # Save JSON
    json_path = os.path.join(args.save_dir, 'robustness_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {json_path}')

    pass  # (pert dirs cleaned in loop)


if __name__ == '__main__':
    main()
