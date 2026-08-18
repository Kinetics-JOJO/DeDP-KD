"""
MONAI-based network architectures with pretrained weights support
Supports VNet, UNETR, and Swin UNETR
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
import logging

# Import MONAI components
try:
    from monai.networks.nets import VNet as MonaiVNet
    from monai.networks.nets import UNETR, SwinUNETR
    from monai.apps import download_and_extract
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    logging.warning("MONAI not installed. Please install with: pip install monai")

# Original VNet from the codebase
from vnet_original import VNet as OriginalVNet


class ModelRegistry:
    """Registry for managing different model architectures"""
    
    # Pretrained model URLs (BraTS weights)
    PRETRAINED_URLS = {
        # Note: These URLs might be outdated. If download fails, please download manually:
        # Swin UNETR: https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR
        'swin_unetr_brats': None,  # Original URL is broken, manual download required
        'unetr_brats': None,  # Original URL might be broken, manual download recommended
        'vnet_brats': None,  # No pretrained weights available, train from scratch
    }
    
    # Alternative download instructions
    DOWNLOAD_INSTRUCTIONS = {
        'swin_unetr_brats': (
            "Please download Swin UNETR pretrained weights manually:\n"
            "1. Visit: https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR\n"
            "2. Or try: wget 'https://drive.google.com/uc?export=download&id=1F6VmCSEUqQlId6QSX7a4Wqh_-eTfxyQl' -O swin_unetr_brats.pt\n"
            "3. Place the file in: {cache_dir}/swin_unetr_brats.pt"
        ),
        'unetr_brats': (
            "Please download UNETR pretrained weights manually:\n"
            "1. Visit MONAI Model Zoo or research-contributions repository\n"
            "2. Place the file in: {cache_dir}/unetr_brats.pt"
        )
    }
    
    @staticmethod
    def get_model(
        model_name: str,
        in_channels: int = 4,
        out_channels: int = 4,
        pretrained: bool = True,
        freeze_encoder: bool = False,
        cache_dir: str = './pretrained_models'
    ) -> nn.Module:
        """
        Get model instance with optional pretrained weights
        
        Args:
            model_name: One of ['vnet', 'unetr', 'swin_unetr']
            in_channels: Number of input channels
            out_channels: Number of output channels
            pretrained: Whether to load pretrained weights
            freeze_encoder: Whether to freeze encoder weights
            cache_dir: Directory to cache pretrained models
        """
        
        if not MONAI_AVAILABLE and model_name != 'vnet':
            raise RuntimeError("MONAI is required for UNETR models. Please install it.")
        
        model_name = model_name.lower()
        
        if model_name == 'vnet':
            # Use original VNet implementation with wrapper
            vnet_model = OriginalVNet(
                n_channels=in_channels,
                n_classes=out_channels,
                n_filters=16,
                normalization='batchnorm'
            )
            model = VNetWrapper(vnet_model)
            
        elif model_name == 'unetr':
            model = UNETRWrapper(
                in_channels=in_channels,
                out_channels=out_channels,
                img_size=(128, 128, 128),
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                proj_type='perceptron',
                norm_name='instance',
                res_block=True,
                dropout_rate=0.0
            )
            
        elif model_name == 'swin_unetr':
            model = SwinUNETRWrapper(
                in_channels=in_channels,
                out_channels=out_channels,
                img_size=(128, 128, 128),
                spatial_dims=3,
                feature_size=48,
                use_checkpoint=False
            )
            
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Load pretrained weights if available
        if pretrained and model_name in ['unetr', 'swin_unetr']:
            model.load_pretrained_weights(cache_dir)
        
        # Freeze encoder if requested
        if freeze_encoder and hasattr(model, 'freeze_encoder'):
            model.freeze_encoder()
        
        return model


class UNETRWrapper(nn.Module):
    """Wrapper for MONAI UNETR with additional functionality"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.model = UNETR(**kwargs)
        self.in_channels = kwargs.get('in_channels', 4)
        self.out_channels = kwargs.get('out_channels', 4)
        
    def forward(self, x):
        """Forward pass returning logits only to avoid double encoder run"""
        logits = self.model(x)
        return None, logits
    
    def get_encoder_features(self, x):
        """Extract features from encoder at different layers"""
        # ViT encoder returns (output, hidden_states_list)
        _, hidden_states = self.model.vit(x)
        return hidden_states
    
    def freeze_encoder(self):
        """Freeze encoder (ViT) parameters"""
        for param in self.model.vit.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters"""
        for param in self.model.vit.parameters():
            param.requires_grad = True
            
    def load_pretrained_weights(self, cache_dir):
        """Load pretrained weights for UNETR"""
        import os
        os.makedirs(cache_dir, exist_ok=True)
        
        weight_path = os.path.join(cache_dir, 'unetr_brats.pt')
        
        if not os.path.exists(weight_path):
            url = ModelRegistry.PRETRAINED_URLS.get('unetr_brats')
            if url:
                try:
                    logging.info("Downloading pretrained UNETR weights...")
                    import urllib.request
                    urllib.request.urlretrieve(url, weight_path)
                    logging.info(f"Downloaded weights to {weight_path}")
                except Exception as e:
                    logging.warning(f"Failed to download pretrained weights: {e}")
                    instructions = ModelRegistry.DOWNLOAD_INSTRUCTIONS.get('unetr_brats', '')
                    if instructions:
                        logging.info("\n" + "="*60)
                        logging.info("MANUAL DOWNLOAD REQUIRED")
                        logging.info("="*60)
                        logging.info(instructions.format(cache_dir=cache_dir))
                        logging.info("="*60 + "\n")
                    logging.warning("Continuing without pretrained weights. Model will be initialized randomly.")
                    return
            else:
                instructions = ModelRegistry.DOWNLOAD_INSTRUCTIONS.get('unetr_brats', '')
                if instructions:
                    logging.info("\n" + "="*60)
                    logging.info("MANUAL DOWNLOAD REQUIRED")
                    logging.info("="*60)
                    logging.info(instructions.format(cache_dir=cache_dir))
                    logging.info("="*60 + "\n")
                logging.warning("No automatic download URL available. Please download weights manually.")
                logging.warning("Continuing without pretrained weights. Model will be initialized randomly.")
                return
        
        if os.path.exists(weight_path):
            try:
                # Fix path separator for cross-platform compatibility
                weight_path_display = weight_path.replace('\\', '/')
                logging.info(f"Loading pretrained UNETR weights from {weight_path_display}...")
                state_dict = torch.load(weight_path, map_location='cpu')
                
                # Handle different state_dict formats
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
                
                # Try to load with strict=False to handle minor mismatches
                incompatible = self.model.load_state_dict(state_dict, strict=False)
                
                if incompatible.missing_keys:
                    logging.info(f"Missing keys in pretrained weights: {len(incompatible.missing_keys)} keys")
                    # This is often normal when the output layer size differs
                    if any('out' in key or 'final' in key or 'decoder' in key for key in incompatible.missing_keys):
                        logging.info("Note: Missing keys in output/decoder layers are expected when num_classes differs from pretrained model")
                if incompatible.unexpected_keys:
                    logging.info(f"Unexpected keys in pretrained weights: {len(incompatible.unexpected_keys)} keys")
                    # This is normal when loading from a model trained with different settings
                    logging.info("Note: Unexpected keys are usually from the pretrained model having different architecture details")
                
                logging.info("Loaded pretrained weights successfully (with expected mismatches for task-specific layers)")
            except Exception as e:
                logging.error(f"Failed to load pretrained weights: {e}")
                logging.warning("Continuing without pretrained weights. Model will be initialized randomly.")
        else:
            logging.warning(f"Pretrained weight file not found at {weight_path}")
            logging.warning("Continuing without pretrained weights. Model will be initialized randomly.")


class SwinUNETRWrapper(nn.Module):
    """Wrapper for MONAI Swin UNETR with additional functionality"""
    
    def __init__(self, **kwargs):
        super().__init__()
        # Ensure img_size is provided for MONAI 1.3.x compatibility
        if 'img_size' not in kwargs:
            kwargs['img_size'] = (96, 96, 96)  # Default size for 3D medical images
            print(f"Warning: img_size not provided, using default {kwargs['img_size']}")
        
        self.model = SwinUNETR(**kwargs)
        self.in_channels = kwargs.get('in_channels', 4)
        self.out_channels = kwargs.get('out_channels', 4)
        
    def forward(self, x):
        """Forward pass returning logits only to avoid double encoder run"""
        logits = self.model(x)
        return None, logits
    
    def get_encoder_features(self, x):
        """Extract features from encoder at different layers"""
        # Swin Transformer encoder forward returns a list of hidden states
        hidden_states_out = self.model.swinViT(x)
        
        return hidden_states_out
    
    def freeze_encoder(self):
        """Freeze encoder (Swin Transformer) parameters"""
        for param in self.model.swinViT.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters"""
        for param in self.model.swinViT.parameters():
            param.requires_grad = True
            
    def load_pretrained_weights(self, cache_dir):
        """Load pretrained weights for Swin UNETR"""
        import os
        os.makedirs(cache_dir, exist_ok=True)
        
        weight_path = os.path.join(cache_dir, 'swin_unetr_brats.pt')
        
        if not os.path.exists(weight_path):
            url = ModelRegistry.PRETRAINED_URLS.get('swin_unetr_brats')
            if url:
                try:
                    logging.info("Downloading pretrained Swin UNETR weights...")
                    import urllib.request
                    urllib.request.urlretrieve(url, weight_path)
                    logging.info(f"Downloaded weights to {weight_path}")
                except Exception as e:
                    logging.warning(f"Failed to download pretrained weights: {e}")
                    instructions = ModelRegistry.DOWNLOAD_INSTRUCTIONS.get('swin_unetr_brats', '')
                    if instructions:
                        logging.info("\n" + "="*60)
                        logging.info("MANUAL DOWNLOAD REQUIRED")
                        logging.info("="*60)
                        logging.info(instructions.format(cache_dir=cache_dir))
                        logging.info("="*60 + "\n")
                    logging.warning("Continuing without pretrained weights. Model will be initialized randomly.")
                    return
            else:
                instructions = ModelRegistry.DOWNLOAD_INSTRUCTIONS.get('swin_unetr_brats', '')
                if instructions:
                    logging.info("\n" + "="*60)
                    logging.info("MANUAL DOWNLOAD REQUIRED")
                    logging.info("="*60)
                    logging.info(instructions.format(cache_dir=cache_dir))
                    logging.info("="*60 + "\n")
                logging.warning("No automatic download URL available. Please download weights manually.")
                logging.warning("Continuing without pretrained weights. Model will be initialized randomly.")
                return
        
        if os.path.exists(weight_path):
            try:
                # Fix path separator for cross-platform compatibility
                weight_path_display = weight_path.replace('\\', '/')
                logging.info(f"Loading pretrained Swin UNETR weights from {weight_path_display}...")
                state_dict = torch.load(weight_path, map_location='cpu')
                
                # Handle different state_dict formats
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
                
                # Try to load with strict=False to handle minor mismatches
                incompatible = self.model.load_state_dict(state_dict, strict=False)
                
                if incompatible.missing_keys:
                    logging.info(f"Missing keys in pretrained weights: {len(incompatible.missing_keys)} keys")
                    # This is often normal when the output layer size differs
                    if any('out' in key or 'final' in key or 'decoder' in key for key in incompatible.missing_keys):
                        logging.info("Note: Missing keys in output/decoder layers are expected when num_classes differs from pretrained model")
                if incompatible.unexpected_keys:
                    logging.info(f"Unexpected keys in pretrained weights: {len(incompatible.unexpected_keys)} keys")
                    # This is normal when loading from a model trained with different settings
                    logging.info("Note: Unexpected keys are usually from the pretrained model having different architecture details")
                
                logging.info("Loaded pretrained weights successfully (with expected mismatches for task-specific layers)")
            except Exception as e:
                logging.error(f"Failed to load pretrained weights: {e}")
                logging.warning("Continuing without pretrained weights. Model will be initialized randomly.")
        else:
            logging.warning(f"Pretrained weight file not found at {weight_path}")
            logging.warning("Continuing without pretrained weights. Model will be initialized randomly.")


class VNetWrapper(nn.Module):
    """Wrapper for original VNet to maintain consistency"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # Preserve original VNet API: return (feature, out)
        feature, out = self.model(x)
        return feature, out
    
    def freeze_encoder(self):
        """Freeze encoder blocks"""
        # Freeze first 4 blocks (encoder)
        for name, param in self.model.named_parameters():
            if any(block in name for block in ['block_one', 'block_two', 'block_three', 'block_four']):
                param.requires_grad = False
                
    def unfreeze_encoder(self):
        """Unfreeze all parameters"""
        for param in self.model.parameters():
            param.requires_grad = True
