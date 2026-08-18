#!/usr/bin/env python3
"""
Script to download pretrained weights for MONAI models
Since the original URLs are broken, this script provides alternative methods
"""

import os
import sys
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def download_with_wget(url, output_path):
    """Download file using wget"""
    cmd = f"wget '{url}' -O '{output_path}'"
    logging.info(f"Executing: {cmd}")
    result = os.system(cmd)
    return result == 0

def download_with_gdown(file_id, output_path):
    """Download from Google Drive using gdown"""
    try:
        import gdown
    except ImportError:
        logging.error("gdown is not installed. Install it with: pip install gdown")
        return False
    
    url = f"https://drive.google.com/uc?id={file_id}"
    logging.info(f"Downloading from Google Drive: {url}")
    gdown.download(url, output_path, quiet=False)
    return os.path.exists(output_path)

def download_swin_unetr_weights(cache_dir):
    """Download Swin UNETR pretrained weights"""
    output_path = os.path.join(cache_dir, 'swin_unetr_brats.pt')
    
    if os.path.exists(output_path):
        logging.info(f"Swin UNETR weights already exist at: {output_path}")
        return True
    
    logging.info("Attempting to download Swin UNETR pretrained weights...")
    
    # Method 1: Try Google Drive link (if available)
    google_drive_id = "1F6VmCSEUqQlId6QSX7a4Wqh_-eTfxyQl"  # Example ID, might need update
    
    logging.info("Method 1: Trying Google Drive download...")
    if download_with_gdown(google_drive_id, output_path):
        logging.info(f"Successfully downloaded to: {output_path}")
        return True
    
    # Method 2: Try wget with direct link
    logging.info("Method 2: Trying wget download...")
    wget_url = f"https://drive.google.com/uc?export=download&id={google_drive_id}"
    if download_with_wget(wget_url, output_path):
        logging.info(f"Successfully downloaded to: {output_path}")
        return True
    
    # If all methods fail, provide manual instructions
    logging.error("Automatic download failed. Please download manually:")
    logging.info("="*60)
    logging.info("MANUAL DOWNLOAD INSTRUCTIONS:")
    logging.info("1. Visit: https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR")
    logging.info("2. Look for pretrained model links in the README")
    logging.info("3. Download the model weights for BraTS or your specific task")
    logging.info(f"4. Save the file as: {output_path}")
    logging.info("="*60)
    
    return False

def download_unetr_weights(cache_dir):
    """Download UNETR pretrained weights"""
    output_path = os.path.join(cache_dir, 'unetr_brats.pt')
    
    if os.path.exists(output_path):
        logging.info(f"UNETR weights already exist at: {output_path}")
        return True
    
    logging.info("UNETR pretrained weights download not implemented yet.")
    logging.info("Please download manually from MONAI Model Zoo:")
    logging.info("https://github.com/Project-MONAI/model-zoo")
    
    return False

def main():
    parser = argparse.ArgumentParser(description='Download pretrained weights for MONAI models')
    parser.add_argument('--model', type=str, choices=['swin_unetr', 'unetr', 'all'], 
                        default='swin_unetr', help='Model type to download weights for')
    parser.add_argument('--cache-dir', type=str, default='./pretrained_models',
                        help='Directory to save pretrained weights')
    
    args = parser.parse_args()
    
    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)
    logging.info(f"Cache directory: {args.cache_dir}")
    
    success = True
    
    if args.model in ['swin_unetr', 'all']:
        if not download_swin_unetr_weights(args.cache_dir):
            success = False
    
    if args.model in ['unetr', 'all']:
        if not download_unetr_weights(args.cache_dir):
            success = False
    
    if success:
        logging.info("Download completed successfully!")
    else:
        logging.warning("Some downloads failed. Please follow manual instructions above.")
        sys.exit(1)

if __name__ == "__main__":
    main()