"""
Evaluation script for Prompt Distillation models
"""

import os
import argparse
import logging
import sys
import numpy as np
import torch
from tqdm import tqdm

from networks_monai import ModelRegistry
from prompt_modules import (
    DualPromptModule,
    PromptedTransformerWrapper,
    ModalityCombination,
    SwinStagePromptModule,
)
from evaluate import test_single_case, evaluate_one_case, convert_to_sitk


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize per-channel image intensities (match training preprocessing)."""
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
    """Map label 4->3 and clamp to [0,3] to align with training."""
    label = label.astype(np.int16, copy=False)
    if np.max(label) > 3:
        label = np.where(label == 4, 3, label)
    return np.clip(label, 0, 3)


def evaluate_prompt_model(args):
    """Evaluate a trained prompt distillation model"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Load checkpoint
    logging.info(f"Loading checkpoint from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # Extract configuration
    config = checkpoint.get('config', {})
    model_type = checkpoint.get('model_type', args.model_type)
    student_modalities = checkpoint.get('student_modalities', [0])
    disable_prompts = bool(config.get('disable_prompts', False))
    
    logging.info(f"Model type: {model_type}")
    logging.info(f"Student modalities: {ModalityCombination.get_combination_key(student_modalities)}")
    
    # Create model
    num_student_channels = len(student_modalities)
    model = ModelRegistry.get_model(
        model_name=model_type,
        in_channels=num_student_channels,
        out_channels=args.num_classes,
        pretrained=False,
        freeze_encoder=False
    )
    
    # Add prompt module if transformer-based and not disabled
    if model_type in ['unetr', 'swin_unetr'] and not disable_prompts:
        if model_type == 'unetr':
            embedding_dim = 768
            num_layers = 12
            prompt_module = DualPromptModule(
                num_layers=num_layers,
                embedding_dim=embedding_dim,
                general_prompt_length=config.get('general_prompt_length', 5),
                expert_prompt_length=config.get('expert_prompt_length', 5)
            )
        else:  # swin_unetr
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
            model_type=model_type
        )
    
    # Load model weights
    model.load_state_dict(checkpoint['student_state_dict'])
    model.cuda()
    model.eval()
    
    # Prepare test data
    test_list = []
    with open(os.path.join(args.data_dir, 'test_list.txt'), 'r') as f:
        for line in f:
            test_list.append(line.strip())
    
    test_paths = [
        os.path.join(args.data_dir, 'brats2020', f"{x}.npy")
        for x in test_list
    ]
    
    logging.info(f"Test set includes {len(test_paths)} samples")
    
    # Evaluation settings
    CROP_SIZE = (128, 128, 128)
    STRIDE = tuple([x // 2 for x in CROP_SIZE])
    
    # Results storage
    dice_arr = []
    hd_arr = []
    sen_arr = []
    spe_arr = []
    
    # Create output directory if saving visualizations
    if args.save_vis:
        os.makedirs(args.output_path, exist_ok=True)
    
    # Evaluate
    logging.info("Starting evaluation...")
    with torch.no_grad():
        for idx, test_path in enumerate(tqdm(test_paths, desc="Evaluating")):
            # Load data
            data = np.load(test_path)
            if not np.isfinite(data).all():
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            image = data[0:4]
            label = data[4]
            image = _normalize_image(image)
            label = _sanitize_label(label)
            
            # Extract student modalities
            student_image = image[student_modalities]
            
            # Predict
            if isinstance(model, PromptedTransformerWrapper):
                def net_fn(x):
                    return model(x, student_modalities)
                predict, score_map = test_single_case(
                    net_fn,
                    student_image,
                    STRIDE,
                    CROP_SIZE,
                    args.num_classes
                )
            else:
                predict, score_map = test_single_case(
                    model,
                    student_image,
                    STRIDE,
                    CROP_SIZE,
                    args.num_classes
                )
            
            # Calculate metrics
            hd, dice, sen, spe = evaluate_one_case(predict, label)
            dice_arr.append(dice)
            hd_arr.append(hd)
            sen_arr.append(sen)
            spe_arr.append(spe)
            
            dice_mean = np.mean(dice)
            logging.info(f"Sample [{idx}] - {os.path.basename(test_path)}: Dice = {dice_mean:.4f}")
            
            # Save visualizations if requested
            if args.save_vis:
                sample_name = os.path.basename(test_path).replace('.npy', '')
                sample_dir = os.path.join(args.output_path, sample_name)
                os.makedirs(sample_dir, exist_ok=True)
                
                # Save images
                for i, mod_idx in enumerate(student_modalities):
                    convert_to_sitk(
                        image[mod_idx],
                        os.path.join(sample_dir, f"image_{i}.nii.gz")
                    )
                
                # Save label and prediction
                convert_to_sitk(
                    label.astype(np.uint8),
                    os.path.join(sample_dir, "label.nii.gz")
                )
                convert_to_sitk(
                    predict.astype(np.uint8),
                    os.path.join(sample_dir, "predict.nii.gz")
                )
    
    # Calculate statistics
    dice_arr = np.array(dice_arr) * 100  # Convert to percentage
    hd_arr = np.array(hd_arr)
    sen_arr = np.array(sen_arr)
    spe_arr = np.array(spe_arr)
    
    dice_mean = np.nanmean(dice_arr, axis=0)
    hd_mean = np.nanmean(hd_arr, axis=0)
    sen_mean = np.nanmean(sen_arr, axis=0)
    spe_mean = np.nanmean(spe_arr, axis=0)
    
    # Save results
    np.save(os.path.join(args.output_path, 'dice_arr.npy'), dice_arr)
    np.save(os.path.join(args.output_path, 'hd_arr.npy'), hd_arr)
    np.save(os.path.join(args.output_path, 'sen_arr.npy'), sen_arr)
    np.save(os.path.join(args.output_path, 'spe_arr.npy'), spe_arr)
    
    # Print results
    logging.info("="*60)
    logging.info("Evaluation Results:")
    logging.info("="*60)
    logging.info(f"Model: {model_type}")
    logging.info(f"Student Modalities: {ModalityCombination.get_combination_key(student_modalities)}")
    logging.info(f"Checkpoint: {args.model_path}")
    logging.info("-"*60)
    logging.info("Statistical indicators on test set (WT/TC/ET):")
    logging.info(f"Dice: [{dice_mean[0]:.2f}, {dice_mean[1]:.2f}, {dice_mean[2]:.2f}]%")
    logging.info(f"HD95: [{hd_mean[0]:.3f}, {hd_mean[1]:.3f}, {hd_mean[2]:.3f}]")
    logging.info(f"Sensitivity: [{sen_mean[0]:.3f}, {sen_mean[1]:.3f}, {sen_mean[2]:.3f}]")
    logging.info(f"Specificity: [{spe_mean[0]:.3f}, {spe_mean[1]:.3f}, {spe_mean[2]:.3f}]")
    logging.info("-"*60)
    logging.info(f"Average Dice: {np.mean(dice_mean):.2f}%")
    logging.info(f"Average HD95: {np.mean(hd_mean):.3f}")
    logging.info(f"Average Sensitivity: {np.mean(sen_mean):.3f}")
    logging.info(f"Average Specificity: {np.mean(spe_mean):.3f}")
    logging.info("="*60)
    
    # Write results to file
    with open(os.path.join(args.output_path, 'results.txt'), 'w') as f:
        f.write(f"Model: {model_type}\n")
        f.write(f"Student Modalities: {ModalityCombination.get_combination_key(student_modalities)}\n")
        f.write(f"Checkpoint: {args.model_path}\n")
        f.write("-"*60 + "\n")
        f.write("Statistical indicators on test set (WT/TC/ET):\n")
        f.write(f"Dice: [{dice_mean[0]:.2f}, {dice_mean[1]:.2f}, {dice_mean[2]:.2f}]%\n")
        f.write(f"HD95: [{hd_mean[0]:.3f}, {hd_mean[1]:.3f}, {hd_mean[2]:.3f}]\n")
        f.write(f"Sensitivity: [{sen_mean[0]:.3f}, {sen_mean[1]:.3f}, {sen_mean[2]:.3f}]\n")
        f.write(f"Specificity: [{spe_mean[0]:.3f}, {spe_mean[1]:.3f}, {spe_mean[2]:.3f}]\n")
        f.write("-"*60 + "\n")
        f.write(f"Average Dice: {np.mean(dice_mean):.2f}%\n")
        f.write(f"Average HD95: {np.mean(hd_mean):.3f}\n")
        f.write(f"Average Sensitivity: {np.mean(sen_mean):.3f}\n")
        f.write(f"Average Specificity: {np.mean(spe_mean):.3f}\n")
    
    logging.info(f"Results saved to {args.output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Prompt Distillation Model')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='swin_unetr',
                        choices=['vnet', 'unetr', 'swin_unetr'],
                        help='Model architecture (will be overridden by checkpoint)')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Path to data directory')
    parser.add_argument('--output_path', type=str, default='../results/prompt_eval',
                        help='Path to save results')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of segmentation classes')
    parser.add_argument('--save_vis', action='store_true',
                        help='Save visualization results')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device to use')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Run evaluation
    evaluate_prompt_model(args)


if __name__ == '__main__':
    main()
