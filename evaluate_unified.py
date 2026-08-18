"""
Unified evaluation script for Prompt Distillation models trained in `unified` mode.

Differences from evaluate_prompt.py:
- Builds the student with in_channels=4 (must match unified-mode training).
- Iterates over all 15 non-empty modality subsets (or a user-specified list) and
  zero-fills missing channels per-subset before forward pass.
- Reports per-subset Dice/HD95/Sen/Spe in a structured table and writes a CSV.

This script is meant for checkpoints saved by training with
    --training_mode unified
The legacy per-subset checkpoints should still be evaluated with evaluate_prompt.py.
"""

import os
import argparse
import logging
import csv
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Optional

from networks_monai import ModelRegistry
from prompt_modules import (
    DualPromptModule,
    PromptedTransformerWrapper,
    ModalityCombination,
    SwinStagePromptModule,
)
from evaluate import test_single_case, evaluate_one_case, convert_to_sitk


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """Per-channel normalization, matching training preprocessing."""
    image = image.astype(np.float32, copy=False)
    for c in range(image.shape[0]):
        ch = image[c]
        ch = np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0)
        mean = float(np.mean(ch))
        std = float(np.std(ch))
        if not np.isfinite(mean):
            mean = 0.0
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        ch = (ch - mean) / std
        ch = np.clip(ch, -5.0, 5.0)
        image[c] = ch
    return image


def _sanitize_label(label: np.ndarray) -> np.ndarray:
    label = label.astype(np.int16, copy=False)
    if np.max(label) > 3:
        label = np.where(label == 4, 3, label)
    return np.clip(label, 0, 3)


def _zero_fill_for_subset(image_4ch: np.ndarray, modality_indices: List[int]) -> np.ndarray:
    """Keep 4 channels; zero out channels NOT in modality_indices."""
    out = image_4ch.copy()
    for c in range(4):
        if c not in modality_indices:
            out[c] = 0.0
    return out


def _parse_subsets_arg(subsets_arg: str) -> List[List[int]]:
    """Parse --subsets into a list of modality index lists.

    Supported values:
      - 'all'  : all 15 non-empty subsets
      - 'size1', 'size2', 'size3', 'size4' : subsets of a specific size
      - comma-separated modality-combination keys, e.g. 'T1,T1+T2,FLAIR+T1ce'
    """
    all_combos = ModalityCombination.get_all_combinations()
    all_subsets = [
        sorted(ModalityCombination.get_indices_from_key('+'.join(c))) for c in all_combos
    ]

    s = subsets_arg.strip().lower()
    if s == 'all':
        return all_subsets

    if s in ('size1', 'size2', 'size3', 'size4'):
        k = int(s[-1])
        return [sub for sub in all_subsets if len(sub) == k]

    result = []
    for item in subsets_arg.split(','):
        item = item.strip()
        if not item:
            continue
        indices = ModalityCombination.get_indices_from_key(item)
        result.append(sorted(indices))
    if not result:
        raise ValueError(f"Could not parse --subsets: {subsets_arg}")
    return result


def _build_unified_model(args, checkpoint):
    """Rebuild the student network the same way the unified trainer built it."""
    config = checkpoint.get('config', {})
    model_type = checkpoint.get('model_type', args.model_type)
    disable_prompts = bool(config.get('disable_prompts', False))
    training_mode = checkpoint.get('training_mode', config.get('training_mode', 'per_subset'))

    if training_mode != 'unified' and not args.force:
        raise RuntimeError(
            f"Checkpoint training_mode is '{training_mode}', not 'unified'. "
            f"Use evaluate_prompt.py for per-subset checkpoints, or pass --force to "
            f"evaluate anyway (the in_channels will be assumed to be 4)."
        )

    # Unified mode -> in_channels must be 4
    model = ModelRegistry.get_model(
        model_name=model_type,
        in_channels=4,
        out_channels=args.num_classes,
        pretrained=False,
        freeze_encoder=False,
    )

    if model_type in ('unetr', 'swin_unetr') and not disable_prompts:
        if model_type == 'unetr':
            embedding_dim = 768
            num_layers = 12
            prompt_module = DualPromptModule(
                num_layers=num_layers,
                embedding_dim=embedding_dim,
                general_prompt_length=config.get('general_prompt_length', 5),
                expert_prompt_length=config.get('expert_prompt_length', 5),
            )
        else:
            feature_size = int(config.get('swin_feature_size', 48))
            stage_dims = [feature_size, feature_size * 2, feature_size * 4, feature_size * 8]
            prompt_module = SwinStagePromptModule(
                stage_dims=stage_dims,
                general_prompt_length=config.get('general_prompt_length', 5),
                expert_prompt_length=config.get('expert_prompt_length', 5),
                general_stages=[0, 1],
                expert_stages=[2, 3],
            )
        model = PromptedTransformerWrapper(
            model=model,
            prompt_module=prompt_module,
            model_type=model_type,
        )

    model.load_state_dict(checkpoint['student_state_dict'])
    model.cuda()
    model.eval()
    return model, model_type, training_mode


def evaluate_unified_model(args):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    logging.info(f"Loading checkpoint from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model, model_type, training_mode = _build_unified_model(args, checkpoint)
    logging.info(f"Model type: {model_type} | training_mode(ckpt): {training_mode}")

    # Resolve subsets to evaluate
    subsets = _parse_subsets_arg(args.subsets)
    logging.info(f"Evaluating on {len(subsets)} modality subsets")

    # Test list
    test_list_path = os.path.join(args.data_dir, args.test_list)
    with open(test_list_path, 'r') as f:
        test_list = [line.strip() for line in f if line.strip()]
    test_paths = [os.path.join(args.data_dir, 'brats2020', f"{x}.npy") for x in test_list]
    logging.info(f"Test set includes {len(test_paths)} samples")

    # Pre-load & cache preprocessed images (to avoid re-normalizing 15 times)
    logging.info("Pre-loading and normalizing test volumes...")
    cache = []
    for p in tqdm(test_paths, desc='Preloading'):
        data = np.load(p)
        if not np.isfinite(data).all():
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        image = _normalize_image(data[0:4])
        label = _sanitize_label(data[4])
        cache.append((os.path.basename(p), image, label))

    CROP_SIZE = (128, 128, 128)
    STRIDE = tuple([x // 2 for x in CROP_SIZE])

    os.makedirs(args.output_path, exist_ok=True)

    # For each subset, evaluate over the full test set
    subset_results = []  # list of dicts per subset
    logging.info("=" * 80)
    logging.info(f"{'Subset':<20} {'WT':>8} {'TC':>8} {'ET':>8} {'Mean':>8}")
    logging.info("-" * 80)

    with torch.no_grad():
        for subset in subsets:
            subset_key = ModalityCombination.get_combination_key(subset)
            dice_arr, hd_arr, sen_arr, spe_arr = [], [], [], []

            for name, image, label in tqdm(cache, desc=subset_key, leave=False):
                student_image = _zero_fill_for_subset(image, subset)

                if isinstance(model, PromptedTransformerWrapper):
                    def net_fn(x, _subset=subset):
                        return model(x, _subset)
                    predict, _ = test_single_case(
                        net_fn, student_image, STRIDE, CROP_SIZE, args.num_classes
                    )
                else:
                    predict, _ = test_single_case(
                        model, student_image, STRIDE, CROP_SIZE, args.num_classes
                    )

                hd, dice, sen, spe = evaluate_one_case(predict, label)
                dice_arr.append(dice)
                hd_arr.append(hd)
                sen_arr.append(sen)
                spe_arr.append(spe)

            dice_arr = np.array(dice_arr) * 100
            hd_arr = np.array(hd_arr)
            sen_arr = np.array(sen_arr)
            spe_arr = np.array(spe_arr)

            dice_mean = np.nanmean(dice_arr, axis=0)
            hd_mean = np.nanmean(hd_arr, axis=0)
            sen_mean = np.nanmean(sen_arr, axis=0)
            spe_mean = np.nanmean(spe_arr, axis=0)

            row = {
                'subset_key': subset_key,
                'size': len(subset),
                'dice_wt': float(dice_mean[0]),
                'dice_tc': float(dice_mean[1]),
                'dice_et': float(dice_mean[2]),
                'dice_avg': float(np.mean(dice_mean)),
                'hd95_wt': float(hd_mean[0]),
                'hd95_tc': float(hd_mean[1]),
                'hd95_et': float(hd_mean[2]),
                'sen_wt': float(sen_mean[0]),
                'sen_tc': float(sen_mean[1]),
                'sen_et': float(sen_mean[2]),
                'spe_wt': float(spe_mean[0]),
                'spe_tc': float(spe_mean[1]),
                'spe_et': float(spe_mean[2]),
            }
            subset_results.append(row)

            logging.info(
                f"{subset_key:<20} "
                f"{row['dice_wt']:>7.2f}  "
                f"{row['dice_tc']:>7.2f}  "
                f"{row['dice_et']:>7.2f}  "
                f"{row['dice_avg']:>7.2f}"
            )

            # Save raw per-case arrays for this subset
            subset_dir = os.path.join(args.output_path, f"subset_{subset_key}")
            os.makedirs(subset_dir, exist_ok=True)
            np.save(os.path.join(subset_dir, 'dice_arr.npy'), dice_arr)
            np.save(os.path.join(subset_dir, 'hd_arr.npy'), hd_arr)
            np.save(os.path.join(subset_dir, 'sen_arr.npy'), sen_arr)
            np.save(os.path.join(subset_dir, 'spe_arr.npy'), spe_arr)

    # Overall averages
    avg_wt = float(np.mean([r['dice_wt'] for r in subset_results]))
    avg_tc = float(np.mean([r['dice_tc'] for r in subset_results]))
    avg_et = float(np.mean([r['dice_et'] for r in subset_results]))
    avg_all = float(np.mean([r['dice_avg'] for r in subset_results]))

    logging.info("-" * 80)
    logging.info(f"{'Average':<20} {avg_wt:>7.2f}  {avg_tc:>7.2f}  {avg_et:>7.2f}  {avg_all:>7.2f}")
    logging.info("=" * 80)

    # Write a CSV summary
    csv_path = os.path.join(args.output_path, 'unified_eval_summary.csv')
    fieldnames = list(subset_results[0].keys())
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in subset_results:
            w.writerow(r)
        # Append an AVG row
        w.writerow({
            'subset_key': 'AVG',
            'size': -1,
            'dice_wt': avg_wt,
            'dice_tc': avg_tc,
            'dice_et': avg_et,
            'dice_avg': avg_all,
            'hd95_wt': float(np.mean([r['hd95_wt'] for r in subset_results])),
            'hd95_tc': float(np.mean([r['hd95_tc'] for r in subset_results])),
            'hd95_et': float(np.mean([r['hd95_et'] for r in subset_results])),
            'sen_wt': float(np.mean([r['sen_wt'] for r in subset_results])),
            'sen_tc': float(np.mean([r['sen_tc'] for r in subset_results])),
            'sen_et': float(np.mean([r['sen_et'] for r in subset_results])),
            'spe_wt': float(np.mean([r['spe_wt'] for r in subset_results])),
            'spe_tc': float(np.mean([r['spe_tc'] for r in subset_results])),
            'spe_et': float(np.mean([r['spe_et'] for r in subset_results])),
        })
    logging.info(f"Per-subset summary written to {csv_path}")

    # Also write a markdown-flavored results.txt for quick pasting into papers
    md_path = os.path.join(args.output_path, 'results_table.md')
    with open(md_path, 'w') as f:
        f.write(f"# Unified Eval Results\n\n")
        f.write(f"- Checkpoint: `{args.model_path}`\n")
        f.write(f"- Model type: `{model_type}`\n")
        f.write(f"- training_mode (ckpt): `{training_mode}`\n")
        f.write(f"- #Subsets evaluated: {len(subset_results)}\n\n")
        f.write("| Subset | Size | Dice_WT | Dice_TC | Dice_ET | Dice_Avg |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in subset_results:
            f.write(
                f"| {r['subset_key']} | {r['size']} | "
                f"{r['dice_wt']:.2f} | {r['dice_tc']:.2f} | "
                f"{r['dice_et']:.2f} | {r['dice_avg']:.2f} |\n"
            )
        f.write(f"| **AVG** | - | **{avg_wt:.2f}** | **{avg_tc:.2f}** | **{avg_et:.2f}** | **{avg_all:.2f}** |\n")
    logging.info(f"Markdown results table written to {md_path}")


def main():
    parser = argparse.ArgumentParser(description='Unified (multi-subset) evaluation for Prompt Distillation')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (should be unified-mode)')
    parser.add_argument('--model_type', type=str, default='swin_unetr',
                        choices=['vnet', 'unetr', 'swin_unetr'],
                        help='Fallback model architecture (overridden by checkpoint)')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--test_list', type=str, default='test_list.txt',
                        help='File under data_dir listing test case IDs (one per line)')
    parser.add_argument('--output_path', type=str, default='../results/prompt_eval_unified')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--subsets', type=str, default='all',
                        help="Which subsets to evaluate: 'all', 'size1'..'size4', "
                             "or a comma-separated list of keys like 'T1,T1+T2'.")
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--force', action='store_true',
                        help='Allow evaluating a non-unified checkpoint with in_channels=4')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.output_path, exist_ok=True)

    evaluate_unified_model(args)


if __name__ == '__main__':
    main()
