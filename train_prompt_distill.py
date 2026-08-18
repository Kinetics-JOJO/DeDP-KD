"""
Main training script for Prompt-based Knowledge Distillation
Supports VNet, UNETR, and Swin UNETR with DualPrompt-style injection
"""

import argparse
import os
import torch
import numpy as np
import random
from typing import List

from trainer_prompt import PromptDistillationTrainer
from prompt_modules import ModalityCombination


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _match_modality_name(name: str) -> int:
    """Case-insensitive lookup of a modality name against MODALITY_INDICES."""
    lower_map = {k.lower(): v for k, v in ModalityCombination.MODALITY_INDICES.items()}
    key = name.strip().lower()
    if key in lower_map:
        return lower_map[key]
    raise ValueError(f"Unknown modality name: {name}")


def parse_modalities(modality_str: str) -> List[int]:
    """Parse modality string to indices"""
    if modality_str.lower() == 'all':
        return [0, 1, 2, 3]  # All modalities
    
    # Parse combinations like "T1+T2" or "0,1"
    if '+' in modality_str:
        # Handle named modalities
        modalities = modality_str.split('+')
        return [_match_modality_name(m) for m in modalities]
    elif ',' in modality_str:
        # Handle numeric indices
        return [int(x.strip()) for x in modality_str.split(',')]
    else:
        # Single modality
        if modality_str.isdigit():
            return [int(modality_str)]
        else:
            return [_match_modality_name(modality_str)]
    
    raise ValueError(f"Invalid modality string: {modality_str}")


def main():
    parser = argparse.ArgumentParser(description='Prompt-based Knowledge Distillation Training')
    
    # Model configuration
    parser.add_argument('--model_type', type=str, default='swin_unetr',
                        choices=['vnet', 'unetr', 'swin_unetr'],
                        help='Model architecture to use')
    # Control student encoder freezing (default: freeze)
    parser.add_argument('--freeze_stu_encoder', dest='freeze_encoder', action='store_true', default=True,
                        help='Freeze student encoder (default)')
    parser.add_argument('--train_stu_encoder', dest='freeze_encoder', action='store_false',
                        help='Train student encoder (do not freeze)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    
    # Student modality configuration
    parser.add_argument('--student_modalities', type=str, default='T1',
                        help='Student modalities (e.g., "T1", "T1+T2", "0,1,2", "all"). '
                             'In per_subset mode, this fixes the training subset. '
                             'In unified mode, this is only used as the default validation '
                             'subset when --unified_val_subsets is not provided.')

    # Training mode: per_subset (legacy, default) vs unified (DualPrompt random sampling)
    parser.add_argument('--training_mode', type=str, default='per_subset',
                        choices=['per_subset', 'unified'],
                        help='per_subset: one training run targets a fixed modality subset '
                             '(legacy behavior). unified: each batch samples a random '
                             'non-empty subset and zero-fills missing modalities, so all '
                             '15 expert prompt pools are trained jointly.')
    parser.add_argument('--unified_sample_strategy', type=str, default='uniform_subset',
                        choices=['uniform_subset', 'uniform_size', 'fixed_one_missing'],
                        help='Modality-subset sampling strategy when --training_mode=unified.')
    parser.add_argument('--unified_val_subsets', type=str, default='default',
                        help='Which subsets to validate on in unified mode. '
                             '"default" = only --student_modalities (fast per-epoch signal); '
                             '"all" = all 15 non-empty subsets (slow but thorough); '
                             'or a comma-separated list of keys, e.g. "T1,T1+T2,FLAIR".')

    # Prompt configuration
    parser.add_argument('--general_prompt_length', type=int, default=5,
                        help='Length of general prompts')
    parser.add_argument('--expert_prompt_length', type=int, default=5,
                        help='Length of expert prompts')
    parser.add_argument('--disable_prompts', action='store_true',
                        help='Disable prompt injection for debugging')

    # Region adaptor configuration
    parser.add_argument('--region_aux_weight', type=float, default=0.0,
                        help='Weight of WT/TC/ET auxiliary losses from adaptor heads (0 to disable)')
    parser.add_argument('--region_adaptor_embedding_dim', type=int, default=128,
                        help='Embedding dim of region adaptor prompts')
    parser.add_argument('--region_adaptor_hidden_dim', type=int, default=128,
                        help='Hidden dim of adaptor FiLM MLP')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--max_epoch', type=int, default=1000,
                        help='Maximum number of epochs')
    
    # Loss weights
    parser.add_argument('--seg_weight', type=float, default=1.0,
                        help='Weight for segmentation loss')
    parser.add_argument('--kd_weight', type=float, default=10.0,
                        help='Weight for knowledge distillation loss')
    parser.add_argument('--temperature', type=float, default=10.0,
                        help='Temperature for knowledge distillation')
    # (keep training simple; KD按标准T^2缩放即可)
    
    # Data configuration
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Path to data directory')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    # (不引入额外前景裁剪选项)
    
    # Experiment configuration
    parser.add_argument('--log_dir', type=str, default='../log/prompt_distill',
                        help='Directory for logs and checkpoints')
    parser.add_argument('--cache_dir', type=str, default='./pretrained_models',
                        help='Directory to cache pretrained models')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    # (沿用现有学习率用法，不额外添加调度选项)
    
    # Resume training
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--ckpt_path', type=str, default='',
                        help='Path to checkpoint for resuming')
    
    # GPU configuration
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device to use')
    parser.add_argument('--detect_anomaly', action='store_true',
                        help='Enable autograd anomaly detection (slow, debug only)')
    
    # Teacher model configuration
    parser.add_argument('--teacher_ckpt_path', type=str, default='',
                        help='Path to pre-trained teacher model checkpoint')
    parser.add_argument('--teacher_model_type', type=str, default='',
                        choices=['', 'vnet', 'unetr', 'swin_unetr'],
                        help='Architecture of the teacher checkpoint (override). If empty, inferred from ckpt or falls back to --model_type')
    # (区域loss按当前训练即刻生效；仅更正缩放与掩码)
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Parse student modalities
    student_modalities = parse_modalities(args.student_modalities)
    print(f"Student will use modalities: {student_modalities}")
    print(f"Modality combination: {ModalityCombination.get_combination_key(student_modalities)}")
    print(f"Training mode: {args.training_mode}")
    if args.training_mode == 'unified':
        print(f"  Unified sampling strategy: {args.unified_sample_strategy}")
        print(f"  Unified validation subsets: {args.unified_val_subsets}")

    # Parse unified validation subsets specification into a form consumable by the trainer
    if args.unified_val_subsets.lower() in ("default", "all"):
        unified_val_subsets = args.unified_val_subsets.lower()
    else:
        # Comma-separated list of modality-combination keys
        unified_val_subsets = [s.strip() for s in args.unified_val_subsets.split(',') if s.strip()]
    
    # Create configuration dictionary
    config = {
        # Model settings
        'model_type': args.model_type,
        'freeze_encoder': args.freeze_encoder,
        'pretrained': args.pretrained,
        'num_classes': 4,  # BraTS has 4 classes (0: background, 1: ET, 2: TC, 3: WT)
        
        # Student configuration
        'student_modalities': student_modalities,

        # Training mode (per_subset legacy vs unified DualPrompt random sampling)
        'training_mode': args.training_mode,
        'unified_sample_strategy': args.unified_sample_strategy,
        'unified_val_subsets': unified_val_subsets,

        # Prompt settings
        'general_prompt_length': args.general_prompt_length,
        'expert_prompt_length': args.expert_prompt_length,
        'disable_prompts': args.disable_prompts,

        # Region adaptor settings
        'region_aux_weight': args.region_aux_weight,
        'region_adaptor_embedding_dim': args.region_adaptor_embedding_dim,
        'region_adaptor_hidden_dim': args.region_adaptor_hidden_dim,
        
        # Training settings
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'max_epoch': args.max_epoch,
        # keep simple scheduler settings (no extra args)
        
        # Loss settings
        'seg_weight': args.seg_weight,
        'kd_weight': args.kd_weight,
        'temperature': args.temperature,
        # KD uses standard T^2 scaling implicitly
        
        # Data settings
        'data_dir': args.data_dir,
        'num_workers': args.num_workers,
        # no extra sampling args
        
        # Experiment settings (append training_mode to log_dir so runs don't collide)
        'log_dir': os.path.join(
            args.log_dir,
            (
                f"{args.model_type}_unified_{args.unified_sample_strategy}"
                if args.training_mode == 'unified'
                else f"{args.model_type}_{ModalityCombination.get_combination_key(student_modalities)}"
            ),
        ),
        'cache_dir': args.cache_dir,
        'seed': args.seed,
        'detect_anomaly': args.detect_anomaly,
        
        # Resume settings
        'resume': args.resume,
        'ckpt_path': args.ckpt_path,
        
        # Teacher model settings
        'teacher_ckpt_path': args.teacher_ckpt_path,
        'teacher_model_type': args.teacher_model_type if args.teacher_model_type else None,
        # region aux starts immediately if weight>0
    }
    
    # Create trainer
    print("\n" + "="*50)
    print("Initializing Prompt Distillation Trainer")
    print("="*50)
    
    trainer = PromptDistillationTrainer(config)
    
    # Start training
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    trainer.train()
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
