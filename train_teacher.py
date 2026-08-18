"""
Train teacher model with all modalities
Supports VNet, UNETR, and Swin UNETR architectures
"""

import os
import sys
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from vnet_original import VNet
from networks_monai import ModelRegistry
from datasets import BraTS
from loss import DiceCeLoss
from evaluate import eval_one_dice, test_single_case
from utils import create_if_not
from cuda_memory_utils import setup_cuda_memory_optimization, print_gpu_memory_usage, clear_gpu_cache


# for BraTS
CROP_SIZE = (128, 128, 128)
STRIDE = tuple([x // 2 for x in list(CROP_SIZE)])


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Fix for cuDNN algorithm error - allow benchmark to find best algorithm
    torch.backends.cudnn.deterministic = False  # Changed to False to allow algorithm search
    torch.backends.cudnn.benchmark = True  # Changed to True to enable algorithm benchmarking
    torch.backends.cudnn.enabled = True  # Ensure cuDNN is enabled


def get_args():
    parser = argparse.ArgumentParser(description='Train Teacher Model with All Modalities')
    
    # Model configuration
    parser.add_argument('--model_type', type=str, default='vnet',
                        choices=['vnet', 'unetr', 'swin_unetr'],
                        help='Model architecture to use')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights for UNETR/Swin-UNETR')
    
    # Training configuration
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="Number of gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)")
    parser.add_argument("--num_cls", type=int, default=4, help="number of classes")
    parser.add_argument("--num_channels", type=int, default=4, help="input channels (all modalities)")
    parser.add_argument("--max_epoch", type=int, default=2000, help="maximum epoch number to train")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    
    # Data configuration
    parser.add_argument("--data_dir", type=str, default="../data", help="dataset path")
    parser.add_argument("--num_workers", type=int, default=4, help="number of data loading workers")
    
    # Logging configuration
    parser.add_argument("--log_dir", type=str, default="../log/teacher", help="log directory")
    parser.add_argument("--save_interval", type=int, default=50, help="save checkpoint every N epochs")
    parser.add_argument("--val_interval", type=int, default=5, help="validation interval")
    
    # Resume training
    parser.add_argument("--resume", action='store_true', help="resume from checkpoint")
    parser.add_argument("--ckpt_path", type=str, default="", help="checkpoint path to resume from")
    
    # Cache directory for pretrained models
    parser.add_argument("--cache_dir", type=str, default="./pretrained_models",
                        help="Directory to cache pretrained models")
    
    args = parser.parse_args()
    return args


def create_model(args):
    """Create model based on architecture type"""
    
    if args.model_type == 'vnet':
        # VNet - no pretrained weights available
        print("Creating VNet model (training from scratch)...")
        model = VNet(
            n_channels=args.num_channels,
            n_classes=args.num_cls,
            n_filters=16,
            normalization="batchnorm"
        )
        
    elif args.model_type in ['unetr', 'swin_unetr']:
        # UNETR or Swin UNETR - can use pretrained weights
        print(f"Creating {args.model_type.upper()} model...")
        
        # Use ModelRegistry to get MONAI models
        model = ModelRegistry.get_model(
            model_name=args.model_type,
            in_channels=args.num_channels,
            out_channels=args.num_cls,
            pretrained=args.pretrained,  # This will download and load pretrained weights
            freeze_encoder=False,  # Don't freeze for teacher training
            cache_dir=args.cache_dir
        )
        
        if args.pretrained:
            print(f"Loaded pretrained weights for {args.model_type.upper()}")
        else:
            print(f"Training {args.model_type.upper()} from scratch")
    
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    return model


def train_epoch(model, train_loader, optimizer, loss_criterion, epoch, writer, iter_num, args):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_dice_loss = 0
    epoch_ce_loss = 0
    
    for idx, (images, labels) in enumerate(train_loader):
        images, labels = images.float().cuda(), labels.cuda()
        
        # Clear GPU cache periodically to prevent memory issues
        if idx % 10 == 0:
            torch.cuda.empty_cache()
        
        # Forward pass with error handling for cuDNN
        try:
            if args.model_type == 'vnet':
                _, logits = model(images)
            else:
                # MONAI models return (features, logits)
                _, logits = model(images)
        except RuntimeError as e:
            if "cuDNN" in str(e) or "out of memory" in str(e):
                # Clear cache and try with reduced memory
                print(f"Warning: cuDNN error encountered. Clearing cache and retrying...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Try again
                if args.model_type == 'vnet':
                    _, logits = model(images)
                else:
                    _, logits = model(images)
            else:
                raise e
        
        # Guard against NaN/Inf in logits before loss
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logging.warning(f"NaN/Inf detected in logits at iteration {idx}. Skipping this batch.")
            optimizer.zero_grad()
            continue

        # Calculate loss
        dice_loss, ce_loss, loss = loss_criterion(logits, labels)
        
        # Check for NaN in loss
        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning(f"NaN or Inf detected in loss at iteration {idx}. Skipping this batch.")
            optimizer.zero_grad()
            continue
        
        # Scale loss for gradient accumulation
        loss = loss / args.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights only after accumulating gradients
        if (idx + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
        
        # Record losses
        epoch_loss += loss.item()
        epoch_dice_loss += dice_loss.item()
        epoch_ce_loss += ce_loss.item()
        
        # Log to tensorboard
        writer.add_scalar("train/loss", loss.item(), iter_num)
        writer.add_scalar("train/dice_loss", dice_loss.item(), iter_num)
        writer.add_scalar("train/ce_loss", ce_loss.item(), iter_num)
        
        iter_num += 1
        
        if idx % 10 == 0:
            logging.info(f"Epoch [{epoch}/{args.max_epoch}] Iter [{idx}/{len(train_loader)}] "
                        f"Loss: {loss.item():.4f}, Dice: {dice_loss.item():.4f}, CE: {ce_loss.item():.4f}")
    
    avg_loss = epoch_loss / len(train_loader)
    avg_dice_loss = epoch_dice_loss / len(train_loader)
    avg_ce_loss = epoch_ce_loss / len(train_loader)
    
    return avg_loss, avg_dice_loss, avg_ce_loss, iter_num


def validate(model, val_list, epoch, writer, args):
    """Validate model performance"""
    model.eval()
    
    dice_wt_sum, dice_co_sum, dice_ec_sum = 0, 0, 0
    
    with torch.no_grad():
        for i, img_path in enumerate(val_list):
            if i >= 5:  # Only validate on first 5 samples for speed
                break
                
            data = np.load(img_path)
            image = data[0:4]  # All 4 modalities
            label = data[4]
            
            # Test with sliding window
            # test_single_case returns (label_map, score_map), we only need label_map
            pred, _ = test_single_case(
                model, image, STRIDE, CROP_SIZE, args.num_cls, 
                model_type=args.model_type
            )
            
            # Calculate Dice scores (now returns 4 values including mean)
            dice_wt, dice_co, dice_ec, _ = eval_one_dice(pred, label)
            dice_wt_sum += dice_wt
            dice_co_sum += dice_co
            dice_ec_sum += dice_ec
    
    num_samples = min(5, len(val_list))
    avg_dice_wt = dice_wt_sum / num_samples
    avg_dice_co = dice_co_sum / num_samples
    avg_dice_ec = dice_ec_sum / num_samples
    avg_dice = (avg_dice_wt + avg_dice_co + avg_dice_ec) / 3
    
    # Log to tensorboard
    writer.add_scalar("val/dice_wt", avg_dice_wt, epoch)
    writer.add_scalar("val/dice_co", avg_dice_co, epoch)
    writer.add_scalar("val/dice_ec", avg_dice_ec, epoch)
    writer.add_scalar("val/dice_avg", avg_dice, epoch)
    
    logging.info(f"Validation - Epoch {epoch}: "
                f"Dice_WT: {avg_dice_wt:.4f}, Dice_TC: {avg_dice_co:.4f}, "
                f"Dice_ET: {avg_dice_ec:.4f}, Avg_Dice: {avg_dice:.4f}")
    
    return avg_dice, avg_dice_wt, avg_dice_co, avg_dice_ec


def main():
    args = get_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Additional CUDA optimizations for memory and cuDNN
    if torch.cuda.is_available():
        # Apply memory optimizations
        setup_cuda_memory_optimization()
        
        # Print GPU info
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        
        # Print initial memory usage
        print_gpu_memory_usage()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Create directories
    snapshot_path = os.path.join(args.log_dir, f"{args.model_type}_teacher")
    create_if_not(snapshot_path)
    save_model_path = os.path.join(snapshot_path, "model")
    create_if_not(save_model_path)
    
    # Setup logging
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S"
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    # Log configuration
    logging.info("="*60)
    logging.info("Teacher Model Training Configuration")
    logging.info("="*60)
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")
    logging.info("="*60)
    
    # Create model
    model = create_model(args)
    model.cuda()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Initialize training state
    start_epoch = 0
    best_epoch = 0
    best_dice = 0
    best_wt = 0
    best_co = 0
    best_ec = 0
    iter_num = 0
    
    # Resume from checkpoint if specified
    if args.resume and args.ckpt_path:
        logging.info(f"Loading checkpoint from {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0)
        best_epoch = checkpoint.get('best_epoch', 0)
        iter_num = checkpoint.get('iter_num', 0)
        logging.info(f"Resumed from epoch {start_epoch}")
    
    # Create data loaders
    train_dataset = BraTS(args.data_dir, crop_size=CROP_SIZE)
    logging.info(f"Training set includes {len(train_dataset)} samples")
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    # Load validation list
    val_list = []
    with open(os.path.join(args.data_dir, "val_list.txt"), 'r') as f:
        for line in f:
            val_list.append(os.path.join(args.data_dir, "brats2020", line.strip() + ".npy"))
    logging.info(f"Validation set includes {len(val_list)} samples")
    # Loss function
    loss_criterion = DiceCeLoss(args.num_cls)
    
    # Tensorboard writer
    writer = SummaryWriter(os.path.join(snapshot_path, "tensorboard"))
    
    # Training loop
    logging.info("="*60)
    logging.info("Starting Training")
    logging.info("="*60)
    
    for epoch in range(start_epoch, args.max_epoch):
        # Adjust learning rate
        curr_lr = args.lr * (1.0 - float(epoch) / float(args.max_epoch)) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr
        
        logging.info(f"\nEpoch {epoch}/{args.max_epoch}, Learning rate: {curr_lr:.6f}")
        
        # Train for one epoch
        time_start = time.time()
        avg_loss, avg_dice_loss, avg_ce_loss, iter_num = train_epoch(
            model, train_loader, optimizer, loss_criterion, epoch, writer, iter_num, args
        )
        time_end = time.time()
        
        logging.info(f"Epoch {epoch} training completed in {time_end-time_start:.2f}s")
        logging.info(f"Average Loss: {avg_loss:.4f}, Dice Loss: {avg_dice_loss:.4f}, CE Loss: {avg_ce_loss:.4f}")
        
        # Validation
        if (epoch + 1) % args.val_interval == 0:
            avg_dice, dice_wt, dice_co, dice_ec = validate(model, val_list, epoch, writer, args)
            
            # Save best model
            if avg_dice > best_dice:
                best_dice = avg_dice
                best_epoch = epoch
                best_wt = dice_wt
                best_co = dice_co
                best_ec = dice_ec
                
                # Save best model
                best_path = os.path.join(save_model_path, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'best_epoch': best_epoch,
                    'dice_wt': best_wt,
                    'dice_co': best_co,
                    'dice_ec': best_ec,
                    'model_type': args.model_type,
                    'iter_num': iter_num
                }, best_path)
                
                logging.info(f"New best model saved at epoch {epoch} with Dice: {best_dice:.4f}")
            
            logging.info(f"Best so far - Epoch: {best_epoch}, Dice: {best_dice:.4f}, "
                        f"WT: {best_wt:.4f}, TC: {best_co:.4f}, ET: {best_ec:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(save_model_path, f"checkpoint_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_dice': best_dice,
                'best_epoch': best_epoch,
                'model_type': args.model_type,
                'iter_num': iter_num
            }, checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Training completed
    logging.info("="*60)
    logging.info("Training Completed!")
    logging.info(f"Best model: Epoch {best_epoch}, Dice: {best_dice:.4f}")
    logging.info(f"Best scores - WT: {best_wt:.4f}, TC: {best_co:.4f}, ET: {best_ec:.4f}")
    logging.info("="*60)
    
    writer.close()


if __name__ == '__main__':
    main()
