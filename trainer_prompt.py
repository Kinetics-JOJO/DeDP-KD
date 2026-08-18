"""
Prompt-based Knowledge Distillation Trainer
"""

import os
import sys
import logging
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Tuple

from networks_monai import ModelRegistry
from prompt_modules import (
    DualPromptModule,
    PromptedTransformerWrapper,
    ModalityCombination,
    RegionAdaptorHeads,
    SwinStagePromptModule,
)
from datasets import BraTS
from loss import DiceCeLoss, softmax_kl_loss
from evaluate import eval_one_dice, test_single_case
from utils import create_if_not


CROP_SIZE = (128, 128, 128)
STRIDE = tuple([x // 2 for x in list(CROP_SIZE)])


class PromptDistillationTrainer(nn.Module):
    """
    Trainer for Prompt-based Knowledge Distillation
    Teacher uses all modalities, Student uses subset with prompts
    """
    
    def __init__(self, cfg: Dict):
        super().__init__()
        
        print("------Prompt Distillation Configs------")
        for k, v in cfg.items():
            print(f"{k}: {v}")
        print("----------------------------------------")
        
        self.cfg = cfg
        self.num_cls = cfg.get("num_classes", 4)
        self.lr = cfg.get("lr", 0.001)
        self.max_epoch = cfg.get("max_epoch", 1000)
        self.model_type = cfg.get("model_type", "swin_unetr")  # vnet, unetr, swin_unetr
        self.freeze_encoder = cfg.get("freeze_encoder", True)
        # Optional: enable PyTorch autograd anomaly detection for NaN/Inf debugging
        if self.cfg.get("detect_anomaly", False):
            try:
                torch.autograd.set_detect_anomaly(True)
                print("[Debug] Autograd anomaly detection enabled (training will be slower)")
            except Exception:
                pass
        
        # KD settings
        self.T = cfg.get("temperature", 10)
        self.kd_weight = cfg.get("kd_weight", 10)
        self.seg_weight = cfg.get("seg_weight", 1.0)
        # Region auxiliary supervision weight (for WT/TC/ET adaptor heads)
        self.region_aux_weight = cfg.get("region_aux_weight", 0.0)
        
        # Training schedule and control knobs (with safe defaults)
        self.grad_accum_steps = max(1, int(cfg.get("grad_accum_steps", 1)))
        self.lr_schedule = str(cfg.get("lr_schedule", "const")).lower()  # const | poly | cosine
        self.lr_warmup_epochs = int(cfg.get("lr_warmup_epochs", 0))
        # Validation cadence
        self.val_interval = max(1, int(cfg.get("val_interval", 5)))
        self.val_start_epoch = max(0, int(cfg.get("val_start_epoch", self.max_epoch // 4)))
        # Aux losses / KD warmup
        self.region_aux_start_epoch = max(0, int(cfg.get("region_aux_start_epoch", 0)))
        self.kd_warmup_epochs = int(cfg.get("kd_warmup_epochs", 0))
        
        # Prompt settings
        self.general_prompt_length = cfg.get("general_prompt_length", 5)
        self.expert_prompt_length = cfg.get("expert_prompt_length", 5)
        self.disable_prompts = cfg.get("disable_prompts", False)
        self.num_transformer_layers = self._get_num_layers()
        
        # Student modality configuration
        self.student_modalities = cfg.get("student_modalities", [0])  # Default to T1 only
        self.student_modality_key = ModalityCombination.get_combination_key(self.student_modalities)

        # Training mode: 'per_subset' (legacy, default) or 'unified' (random modality sampling)
        # - per_subset: one training run targets a fixed modality subset; only one expert
        #               prompt pool is updated; student takes only the selected channels.
        # - unified:   every batch samples a random non-empty modality subset; the student
        #              always takes 4 channels (missing zero-filled); all 15 expert prompt
        #              pools are trained jointly. This is the setup compatible with the
        #              DualPrompt pool mechanism.
        self.training_mode = str(cfg.get("training_mode", "per_subset")).lower()
        if self.training_mode not in ("per_subset", "unified"):
            raise ValueError(
                f"training_mode must be 'per_subset' or 'unified', got '{self.training_mode}'"
            )
        self.unified_random_sampling = (self.training_mode == "unified")
        # Optional curriculum knob for unified mode:
        # 'uniform_subset'    - uniform over all 15 non-empty subsets (default)
        # 'uniform_size'      - uniform over subset size in [1,4], then uniform within size
        # 'fixed_one_missing' - always drop exactly one modality (easier warmup)
        self.unified_sample_strategy = str(cfg.get(
            "unified_sample_strategy", "uniform_subset"
        )).lower()

        # Cache list of all possible modality index subsets once (used only in unified mode)
        self._all_modality_subsets: List[List[int]] = []
        for combo in ModalityCombination.get_all_combinations():
            self._all_modality_subsets.append(
                sorted(ModalityCombination.get_indices_from_key('+'.join(combo)))
            )

        # Initialize models
        self._init_models()
        
        # Initialize optimizer (only for trainable parameters)
        self._init_optimizer()
        
        # Initialize data loaders
        self._init_dataloaders()
        
        # Loss functions
        self.dice_ce_loss = DiceCeLoss(self.num_cls)
        
        # Logging
        self._init_logging()
        
    def _get_num_layers(self) -> int:
        """Get number of transformer layers based on model type"""
        if self.model_type == 'unetr':
            return 12  # Standard ViT has 12 layers
        elif self.model_type == 'swin_unetr':
            return 4  # Swin Transformer typically has 4 stages
        else:
            return 0  # VNet doesn't have transformer layers
    
    def _init_models(self):
        """Initialize teacher and student models with prompts"""
        
        # Teacher model (uses all 4 modalities)
        print("Initializing teacher model...")
        
        # Check if we have a custom teacher checkpoint
        teacher_ckpt_path = self.cfg.get("teacher_ckpt_path", None)
        
        if teacher_ckpt_path and os.path.exists(teacher_ckpt_path):
            # Load custom trained teacher model (respect the architecture used to train the checkpoint)
            print(f"Loading teacher model from checkpoint: {teacher_ckpt_path}")
            
            # Load checkpoint on CPU to inspect metadata
            checkpoint = torch.load(teacher_ckpt_path, map_location='cpu')
            
            # Determine teacher architecture priority: CLI override -> ckpt metadata -> current model_type
            teacher_model_type = self.cfg.get("teacher_model_type", None)
            if not teacher_model_type:
                teacher_model_type = checkpoint.get("model_type", self.model_type)
            teacher_model_type = str(teacher_model_type).lower()
            
            if teacher_model_type == 'vnet':
                # Build original VNet to match state dict saved during teacher training
                from vnet_original import VNet as OriginalVNet
                self.teacher_model = OriginalVNet(
                    n_channels=4,
                    n_classes=self.num_cls,
                    n_filters=16,
                    normalization="batchnorm"
                )
            else:
                # Build MONAI-based wrapper model to match UNETR/Swin UNETR checkpoints
                self.teacher_model = ModelRegistry.get_model(
                    model_name=teacher_model_type,
                    in_channels=4,
                    out_channels=self.num_cls,
                    pretrained=False,  # loading from checkpoint
                    freeze_encoder=self.freeze_encoder,
                    cache_dir=self.cfg.get("cache_dir", "./pretrained_models")
                )
            
            # Load weights (handle both raw and wrapped state_dict formats)
            state_dict = checkpoint.get('state_dict', checkpoint)
            try:
                incompatible = self.teacher_model.load_state_dict(state_dict, strict=False)
                # Optional: log minimal info about missing/unexpected keys
                missing = getattr(incompatible, 'missing_keys', []) if incompatible is not None else []
                unexpected = getattr(incompatible, 'unexpected_keys', []) if incompatible is not None else []
                if missing:
                    print(f"[Teacher] Missing keys when loading: {len(missing)} (expected for task-specific heads)")
                if unexpected:
                    print(f"[Teacher] Unexpected keys when loading: {len(unexpected)}")
            except Exception as e:
                print(f"Failed to load teacher checkpoint strictly due to: {e}")
                raise
            
            print(f"Successfully loaded teacher model ({teacher_model_type}) from {teacher_ckpt_path}")
        else:
            # Fallback: teacher checkpoint not provided or not found
            resolved = os.path.abspath(teacher_ckpt_path) if teacher_ckpt_path else None
            print(
                f"Using default teacher model initialization... (teacher_ckpt_path: {teacher_ckpt_path}, "
                f"abs: {resolved}, exists: {os.path.exists(teacher_ckpt_path) if teacher_ckpt_path else False})"
            )
            self.teacher_model = ModelRegistry.get_model(
                model_name=self.model_type,
                in_channels=4,  # All modalities
                out_channels=self.num_cls,
                pretrained=True,
                freeze_encoder=self.freeze_encoder,
                cache_dir=self.cfg.get("cache_dir", "./pretrained_models")
            )
        
        self.teacher_model.cuda()
        self.teacher_model.eval()  # Teacher is always in eval mode
        
        # Freeze teacher completely
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Student model (uses subset of modalities)
        # In unified mode, the student must be able to ingest any missing pattern, so
        # we always expose 4 input channels and zero-fill the missing ones at runtime.
        # In per_subset mode we keep the legacy behavior (in_channels == |subset|).
        print("Initializing student model...")
        if self.unified_random_sampling:
            num_student_channels = 4
            print(f"[unified] Student in_channels = 4 (zero-fill for missing modalities)")
        else:
            num_student_channels = len(self.student_modalities)
            print(
                f"[per_subset] Student in_channels = {num_student_channels} "
                f"for modalities {self.student_modality_key}"
            )
        self.student_model = ModelRegistry.get_model(
            model_name=self.model_type,
            in_channels=num_student_channels,
            out_channels=self.num_cls,
            pretrained=True,
            freeze_encoder=self.freeze_encoder,
            cache_dir=self.cfg.get("cache_dir", "./pretrained_models")
        )
        
        # Initialize prompt module if using transformer-based model
        if (self.model_type in ['unetr', 'swin_unetr']) and (not self.disable_prompts):
            print("Initializing prompt module...")
            
            # Determine embedding dimension based on model
            if self.model_type == 'unetr':
                embedding_dim = 768  # ViT-Base
                self.prompt_module = DualPromptModule(
                    num_layers=self.num_transformer_layers,
                    embedding_dim=embedding_dim,
                    general_prompt_length=self.general_prompt_length,
                    expert_prompt_length=self.expert_prompt_length,
                    general_layers=list(range(self.num_transformer_layers // 2)),
                    expert_layers=list(range(self.num_transformer_layers // 2, self.num_transformer_layers))
                )
            else:  # swin_unetr
                feature_size = int(self.cfg.get("swin_feature_size", 48))
                stage_dims = [feature_size, feature_size * 2, feature_size * 4, feature_size * 8]
                self.prompt_module = SwinStagePromptModule(
                    stage_dims=stage_dims,
                    general_prompt_length=self.general_prompt_length,
                    expert_prompt_length=self.expert_prompt_length,
                    general_stages=[0, 1],
                    expert_stages=[2, 3],
                )
            
            # Wrap student model with prompt injection
            self.student_model = PromptedTransformerWrapper(
                model=self.student_model,
                prompt_module=self.prompt_module,
                model_type=self.model_type
            )

            # Expert prompt pool activation policy depends on training_mode:
            # - per_subset: freeze all pools except the one matching student_modality_key
            #               (legacy behavior; matches ACN-style specialist training).
            # - unified:    leave every expert pool trainable so random modality sampling
            #               updates all 15 experts jointly (true DualPrompt pool usage).
            try:
                if self.unified_random_sampling:
                    # Ensure all expert pools (and general pool) are trainable.
                    if hasattr(self.prompt_module, "expert_prompt_pools"):
                        pools_iter = (
                            self.prompt_module.expert_prompt_pools.items()
                            if isinstance(self.prompt_module.expert_prompt_pools, torch.nn.ModuleDict)
                            else [
                                (f"stage_{si}_{k}", p)
                                for si, stage_pools in enumerate(self.prompt_module.expert_prompt_pools)
                                for k, p in stage_pools.items()
                            ]
                        )
                        n_pools = 0
                        for _, pool in pools_iter:
                            for param in pool.parameters():
                                param.requires_grad = True
                            n_pools += 1
                        print(f"[unified] All expert prompt pools enabled for training (count={n_pools})")
                else:
                    active_key = self.student_modality_key
                    if hasattr(self.prompt_module, "set_active_expert_key"):
                        self.prompt_module.set_active_expert_key(active_key)
                    else:
                        for key, pool in self.prompt_module.expert_prompt_pools.items():
                            requires = (key == active_key)
                            for param in pool.parameters():
                                param.requires_grad = requires
                    print(f"[per_subset] Enabled training for expert prompt pool: {active_key}; others frozen")
            except Exception as e:
                print(f"Warning: failed to set expert prompt pool activation policy: {e}")
        
        self.student_model.cuda()

        # Initialize lightweight region adaptor heads operating on final logits
        # They modulate class logits and produce WT/TC/ET auxiliary logits
        self.region_adaptor = RegionAdaptorHeads(
            in_channels=self.num_cls,
            embedding_dim=self.cfg.get('region_adaptor_embedding_dim', 128),
            num_regions=3,
            hidden_dim=self.cfg.get('region_adaptor_hidden_dim', 128)
        ).cuda()
        
        # Freeze encoder if specified
        if self.freeze_encoder and hasattr(self.student_model, 'freeze_encoder'):
            self.student_model.freeze_encoder()
            print("Froze student encoder")
    
    def _init_optimizer(self):
        """Initialize optimizer for trainable parameters only"""
        trainable_params = []
        
        # Add student model trainable parameters
        for name, param in self.student_model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                print(f"Trainable: {name}")

        # Add region adaptor parameters
        for name, param in self.region_adaptor.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                print(f"Trainable: region_adaptor.{name}")
        
        print(f"Total trainable parameters: {len(trainable_params)}")
        
        self.optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.lr,
            weight_decay=self.cfg.get("weight_decay", 1e-5)
        )
    
    def _init_dataloaders(self):
        """Initialize training and validation data loaders"""
        # Training dataset
        train_dataset = BraTS(self.cfg.get("data_dir", "../data"), crop_size=CROP_SIZE, return_id=True)
        print(f"Training set includes {len(train_dataset)} samples")
        
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.cfg.get("batch_size", 4),
            shuffle=True,
            num_workers=self.cfg.get("num_workers", 4),
            pin_memory=True
        )
        
        # Validation list
        val_list = []
        val_list_path = os.path.join(self.cfg.get("data_dir", "../data"), "val_list.txt")
        with open(val_list_path, 'r') as f:
            for line in f:
                val_list.append(line.strip())
        
        self.val_list = [
            os.path.join(self.cfg.get("data_dir", "../data"), "brats2020", f"{x}.npy")
            for x in val_list
        ]
        print(f"Validation set includes {len(self.val_list)} samples")
    
    def _init_logging(self):
        """Initialize logging and tensorboard"""
        snapshot_path = self.cfg.get("log_dir", "../log/prompt_distill")
        create_if_not(snapshot_path)
        
        self.save_model_path = os.path.join(snapshot_path, "model")
        create_if_not(self.save_model_path)
        
        # Setup logging: ensure both file and console output, even if logging was pre-configured
        log_file_path = os.path.join(snapshot_path, "log.txt")
        file_handler = logging.FileHandler(log_file_path, mode='a')
        stream_handler = logging.StreamHandler(sys.stdout)
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s.%(msecs)03d] %(message)s",
            datefmt="%H:%M:%S",
            handlers=[file_handler, stream_handler],
            force=True,
        )
        
        # Tensorboard writer
        self.writer = SummaryWriter(os.path.join(snapshot_path, "tensorboard"))
        
        # Training state
        self.iter_num = 0
        self.start_epoch = 0
        self.best_epoch = 0
        self.best_dice = 0
        self.best_wt = 0
        self.best_co = 0
        self.best_ec = 0
        
        # Load checkpoint if resuming
        if self.cfg.get("resume", False) and self.cfg.get("ckpt_path"):
            self._load_checkpoint(self.cfg["ckpt_path"])
    
    def _load_checkpoint(self, ckpt_path: str):
        """Load checkpoint for resuming training"""
        logging.info(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path)
        
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_dice = ckpt.get("best_dice", 0)
        self.best_epoch = ckpt.get("best_epoch", 0)
        
        # Load model state
        if "student_state_dict" in ckpt:
            self.student_model.load_state_dict(ckpt["student_state_dict"])
        # Load adaptor state if present
        if "region_adaptor_state_dict" in ckpt:
            try:
                self.region_adaptor.load_state_dict(ckpt["region_adaptor_state_dict"])
                logging.info("Loaded region adaptor state from checkpoint")
            except Exception as e:
                logging.warning(f"Failed to load region adaptor state: {e}")
        
        # Load optimizer state
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        
        logging.info(f"Resumed from epoch {self.start_epoch}")

    def _ensure_runtime_defaults(self):
        """Ensure newly introduced attributes exist with safe defaults.

        This protects against AttributeError when loading objects created with
        older code versions or pickled states that bypassed __init__.
        """
        defaults = {
            'grad_accum_steps': 1,
            'lr_schedule': 'const',
            'lr_warmup_epochs': 0,
            'val_interval': 5,
            'val_start_epoch': max(0, self.max_epoch // 4 if hasattr(self, 'max_epoch') else 0),
            'region_aux_start_epoch': 0,
            'kd_warmup_epochs': 0,
        }
        for name, value in defaults.items():
            if not hasattr(self, name) or getattr(self, name) is None:
                try:
                    setattr(self, name, value)
                except Exception:
                    # As a last resort, skip setting if object is frozen
                    pass

    def _compute_kd_weight(self, epoch: int) -> float:
        """Linearly warm up KD weight up to configured kd_weight."""
        if self.kd_warmup_epochs and epoch < self.kd_warmup_epochs:
            # Epoch is 0-indexed; add 1 so the very first epoch gets a small but nonzero weight
            return float(self.kd_weight) * float(epoch + 1) / float(self.kd_warmup_epochs)
        return float(self.kd_weight)
    
    def _sample_modality_subset(self) -> List[int]:
        """Sample a non-empty modality subset for the current unified-mode batch.

        Strategy is controlled by self.unified_sample_strategy:
          - 'uniform_subset'   : pick one of the 15 non-empty subsets uniformly.
          - 'uniform_size'     : first sample size k~Uniform[1..4], then subset of that size.
          - 'fixed_one_missing': always drop exactly one modality (size=3).
        """
        strategy = self.unified_sample_strategy
        if strategy == "uniform_subset":
            return list(random.choice(self._all_modality_subsets))
        if strategy == "uniform_size":
            k = random.randint(1, 4)
            return sorted(random.sample([0, 1, 2, 3], k))
        if strategy == "fixed_one_missing":
            drop = random.randint(0, 3)
            return [i for i in [0, 1, 2, 3] if i != drop]
        # Fallback
        return list(random.choice(self._all_modality_subsets))

    def _prepare_student_input(
        self,
        images: torch.Tensor,
        modality_indices: List[int],
    ) -> torch.Tensor:
        """Prepare student-facing input according to the active training mode.

        - per_subset mode:  slice channels (keeps legacy semantics, in_channels=|subset|).
        - unified mode:     keep 4 channels; zero out channels NOT in modality_indices.
        """
        if self.unified_random_sampling:
            # Zero-fill missing channels, keep 4-channel input
            mask = torch.zeros(4, dtype=images.dtype, device=images.device)
            for c in modality_indices:
                if 0 <= c < 4:
                    mask[c] = 1.0
            mask = mask.view(1, 4, 1, 1, 1)
            return images * mask
        # Legacy per-subset slicing
        return images[:, modality_indices]

    def forward(
        self,
        x: torch.Tensor,
        modality_indices: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through student model.

        modality_indices:
          - If None (default), fall back to self.student_modalities. This preserves
            backward compatibility with validate() / evaluate_prompt.py.
          - If a list is provided (e.g., per-batch random sample in unified mode or
            per-subset enumeration at eval time), it is forwarded to the prompt
            wrapper so the correct E-prompt pool is selected.
        """
        if modality_indices is None:
            modality_indices = self.student_modalities
        if isinstance(self.student_model, PromptedTransformerWrapper):
            return self.student_model(x, modality_indices)
        # Standard forward for plain models/wrappers (e.g., UNETRWrapper/SwinUNETRWrapper)
        return self.student_model(x)
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.student_model.train()
        epoch_loss = 0
        epoch_seg_loss = 0
        epoch_kd_loss = 0
        # Track how often each modality subset is sampled in unified mode (sanity check)
        subset_hits: Dict[str, int] = {}
        # Epoch-specific KD weight (supports warmup)
        kd_weight_epoch = self._compute_kd_weight(epoch)

        time_start = time.time()
        
        for idx, batch in enumerate(self.train_loader):
            # Support (image, label) or (image, label, id)
            if len(batch) == 3:
                images, labels, case_id = batch
            else:
                images, labels = batch
                case_id = None
            images, labels = images.float().cuda(), labels.cuda()

            # Basic data validity checks
            if torch.isnan(images).any() or torch.isinf(images).any():
                logging.warning(f"NaN/Inf in input images at iter {idx}. Skipping batch.")
                continue
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                logging.warning(f"NaN/Inf in labels at iter {idx}. Skipping batch.")
                continue
            
            # Determine active modality subset for this batch
            if self.unified_random_sampling:
                current_modalities = self._sample_modality_subset()
                key = ModalityCombination.get_combination_key(current_modalities)
                subset_hits[key] = subset_hits.get(key, 0) + 1
            else:
                current_modalities = self.student_modalities

            # Prepare student input (slice in per_subset mode; zero-fill in unified mode)
            student_input = self._prepare_student_input(images, current_modalities)

            # Teacher forward (always receives full 4 modalities; teacher is frozen)
            with torch.no_grad():
                teacher_features, teacher_logits = self.teacher_model(images)
                # Guard against invalid teacher outputs
                if torch.isnan(teacher_logits).any() or torch.isinf(teacher_logits).any():
                    logging.warning(f"NaN/Inf in teacher logits at iter {idx} (case: {case_id}). Skipping batch.")
                    continue

            # Student forward (subset modalities with prompts; modality_indices drives the E-prompt lookup)
            student_features, student_logits = self.forward(
                student_input, modality_indices=current_modalities
            )
            # Guard against invalid student outputs
            if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
                logging.warning(f"NaN/Inf in student logits at iter {idx} (case: {case_id}). Skipping batch.")
                continue
            # Ensure shapes match for KD
            if student_logits.shape != teacher_logits.shape:
                logging.warning(
                    f"Logit shape mismatch (student {tuple(student_logits.shape)} vs teacher {tuple(teacher_logits.shape)}) at iter {idx} (case: {case_id}). Skipping batch."
                )
                continue
            
            # Segmentation loss (4-class)
            dice_loss, ce_loss, seg_loss = self.dice_ce_loss(student_logits, labels)

            # Guard against invalid seg loss
            if not torch.isfinite(seg_loss):
                logging.warning(
                    f"Non-finite seg loss at iter {idx} (case: {case_id}). seg={seg_loss}, dice={dice_loss}, ce={ce_loss}. Skipping batch."
                )
                continue

            # Region auxiliary losses (WT/TC/ET) using adaptor heads over student logits
            region_loss = torch.tensor(0.0, device=student_logits.device)
            if self.region_aux_weight > 0 and epoch >= self.region_aux_start_epoch:
                logits_wt, logits_tc, logits_et = self.region_adaptor(student_logits)
                # Build binary targets from 4-class labels
                # labels shape [N,1,D,H,W]
                with torch.no_grad():
                    y = labels[:, 0]
                    target_wt = (y != 0).float().unsqueeze(1)
                    # Robust ET: treat class 3 (or 4) as ET
                    et_mask = ((y == 3) | (y == 4))
                    target_et = et_mask.float().unsqueeze(1)
                    # Tumor core (TC): NCR/NET (1) plus ET
                    target_tc = ((y == 1) | et_mask).float().unsqueeze(1)

                bce = torch.nn.BCEWithLogitsLoss()
                from loss import dice_loss as binary_dice
                prob_wt = torch.sigmoid(logits_wt)
                prob_tc = torch.sigmoid(logits_tc)
                prob_et = torch.sigmoid(logits_et)

                dice_wt = binary_dice(prob_wt, target_wt)
                dice_tc = binary_dice(prob_tc, target_tc)
                dice_et = binary_dice(prob_et, target_et)

                bce_wt = bce(logits_wt, target_wt)
                bce_tc = bce(logits_tc, target_tc)
                bce_et = bce(logits_et, target_et)

                # Average six components equally
                region_loss = (dice_wt + dice_tc + dice_et + bce_wt + bce_tc + bce_et) / 6.0
            
            # Knowledge distillation loss
            kd_loss = softmax_kl_loss(
                student_logits / self.T,
                teacher_logits / self.T,
                reduction="mean"
            ) * (self.T ** 2)

            # Guard against invalid kd loss
            if not torch.isfinite(kd_loss):
                logging.warning(
                    f"Non-finite KD loss at iter {idx} (case: {case_id}). Skipping batch."
                )
                continue
            
            # Total loss
            total_loss = self.seg_weight * seg_loss + kd_weight_epoch * kd_loss + self.region_aux_weight * region_loss

            # Backward and optimize with optional gradient accumulation
            if not torch.isfinite(total_loss):
                # Dump quick diagnostics to help locate culprit
                with torch.no_grad():
                    t_min, t_max, t_mean = teacher_logits.min().item(), teacher_logits.max().item(), teacher_logits.mean().item()
                    s_min, s_max, s_mean = student_logits.min().item(), student_logits.max().item(), student_logits.mean().item()
                    i_min, i_max, i_mean = images.min().item(), images.max().item(), images.mean().item()
                logging.warning(
                    f"Non-finite total loss at iter {idx} (case: {case_id}). Stats: "
                    f"img[min={i_min:.4f}, max={i_max:.4f}, mean={i_mean:.4f}], "
                    f"teacher[min={t_min:.4f}, max={t_max:.4f}, mean={t_mean:.4f}], "
                    f"student[min={s_min:.4f}, max={s_max:.4f}, mean={s_mean:.4f}]"
                )
                continue

            # Prepare optimizer step boundaries
            if idx % self.grad_accum_steps == 0:
                self.optimizer.zero_grad()

            # Scale loss for accumulation
            step_loss = total_loss / float(self.grad_accum_steps)
            step_loss.backward()

            # Step when reaching accumulation boundary or the last batch
            do_step = ((idx + 1) % self.grad_accum_steps == 0) or (idx + 1 == len(self.train_loader))
            if do_step:
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.student_model.parameters() if p.requires_grad] +
                    [p for p in self.region_adaptor.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                self.optimizer.step()
            
            # Logging
            epoch_loss += float(total_loss.item())
            epoch_seg_loss += float(seg_loss.item())
            epoch_kd_loss += float(kd_loss.item())
            
            self.iter_num += 1
            
            # Tensorboard logging
            self.writer.add_scalar("train/loss", total_loss, self.iter_num)
            self.writer.add_scalar("train/seg_loss", seg_loss, self.iter_num)
            self.writer.add_scalar("train/kd_loss", kd_loss, self.iter_num)
            self.writer.add_scalar("train/dice_loss", dice_loss, self.iter_num)
            self.writer.add_scalar("train/ce_loss", ce_loss, self.iter_num)
            if self.region_aux_weight > 0:
                self.writer.add_scalar("train/region_loss", region_loss, self.iter_num)
            
            if idx % 10 == 0:
                # Log KD as its weighted contribution so it reflects effective impact on total loss
                kd_contrib = kd_weight_epoch * kd_loss
                logging.info(
                    f"Epoch [{epoch}/{self.max_epoch}], Iter [{idx}/{len(self.train_loader)}], "
                    f"Loss: {total_loss:.4f}, Seg: {seg_loss:.4f}, KD: {kd_contrib:.4f}"
                )
        
        time_end = time.time()
        epoch_time = (time_end - time_start) / 60
        
        avg_loss = epoch_loss / len(self.train_loader)
        avg_seg_loss = epoch_seg_loss / len(self.train_loader)
        avg_kd_loss = epoch_kd_loss / len(self.train_loader)
        # Report KD as weighted contribution for epoch averages as well
        avg_kd_contrib = kd_weight_epoch * avg_kd_loss
        
        logging.info(
            f"Epoch {epoch} training time: {epoch_time:.2f} minutes, "
            f"Avg Loss: {avg_loss:.4f}, Avg Seg: {avg_seg_loss:.4f}, Avg KD: {avg_kd_contrib:.4f}"
        )

        # Unified-mode sanity log: how many times each subset was sampled this epoch
        if self.unified_random_sampling and subset_hits:
            hits_sorted = sorted(subset_hits.items(), key=lambda kv: -kv[1])
            top_preview = ", ".join(f"{k}:{v}" for k, v in hits_sorted[:5])
            logging.info(
                f"[unified] subsets sampled this epoch = {len(subset_hits)}/15 "
                f"(top5: {top_preview})"
            )
            # Tensorboard: log per-subset count as a scalar namespace so you can eyeball uniformity
            for k, v in subset_hits.items():
                self.writer.add_scalar(f"train_subset_hits/{k}", v, epoch)

        return avg_loss
    
    def _predict_one_case(
        self,
        image_4ch: np.ndarray,
        modality_indices: List[int],
    ) -> np.ndarray:
        """Run inference for a single validation/test volume with given modalities.

        image_4ch: float array [4, D, H, W] (full 4 modalities already loaded)
        modality_indices: which modalities are present for this forward pass
        Returns: predict map with shape [D, H, W]
        """
        # Prepare the input according to training_mode so channel count matches the model
        if self.unified_random_sampling:
            # Keep 4 channels, zero-fill missing ones
            student_image = image_4ch.copy()
            for c in range(4):
                if c not in modality_indices:
                    student_image[c] = 0.0
        else:
            student_image = image_4ch[modality_indices]

        if isinstance(self.student_model, PromptedTransformerWrapper):
            def net_fn(x):
                return self.student_model(x, modality_indices)
            predict, _ = test_single_case(
                net_fn, student_image, STRIDE, CROP_SIZE, self.num_cls
            )
        else:
            predict, _ = test_single_case(
                self.student_model, student_image, STRIDE, CROP_SIZE, self.num_cls
            )
        return predict

    def validate(self, epoch: int):
        """Validate model performance.

        - per_subset mode: evaluate on self.student_modalities (legacy behavior).
        - unified mode:    by default still evaluate on self.student_modalities for fast
                           per-epoch signal, unless cfg['unified_val_subsets'] is provided
                           to enumerate additional subsets (e.g. 'all' or a list of keys).
                           The checkpointing criterion remains a single 'mean Dice' number
                           (the average over whatever subsets were evaluated).
        """
        self.student_model.eval()

        # Determine which subsets to validate on
        val_subsets: List[List[int]]
        if self.unified_random_sampling:
            val_sel = self.cfg.get("unified_val_subsets", None)
            if val_sel is None or val_sel == "default":
                val_subsets = [list(self.student_modalities)]
            elif isinstance(val_sel, str) and val_sel.lower() == "all":
                val_subsets = [list(s) for s in self._all_modality_subsets]
            elif isinstance(val_sel, (list, tuple)):
                # Treat as list of modality-key strings or index-lists
                val_subsets = []
                for item in val_sel:
                    if isinstance(item, str):
                        val_subsets.append(sorted(ModalityCombination.get_indices_from_key(item)))
                    else:
                        val_subsets.append(sorted(list(item)))
            else:
                val_subsets = [list(self.student_modalities)]
        else:
            val_subsets = [list(self.student_modalities)]

        per_subset_scores: Dict[str, Tuple[float, float, float, float]] = {}

        logging.info(f"Starting validation for epoch {epoch} on {len(val_subsets)} subset(s)")
        time_start = time.time()

        with torch.no_grad():
            for subset in val_subsets:
                subset_key = ModalityCombination.get_combination_key(subset)
                dice_all_wt, dice_all_co, dice_all_ec, dice_all_mean = [], [], [], []

                for idx, val_path in enumerate(self.val_list):
                    data = np.load(val_path)
                    image = data[0:4]
                    label = data[4]

                    predict = self._predict_one_case(image, subset)

                    dice_wt, dice_co, dice_ec, dice_mean = eval_one_dice(predict, label)
                    dice_all_wt.append(dice_wt)
                    dice_all_co.append(dice_co)
                    dice_all_ec.append(dice_ec)
                    dice_all_mean.append(dice_mean)

                    if idx % 5 == 0 and len(val_subsets) == 1:
                        logging.info(f"[{subset_key}] Sample [{idx}/{len(self.val_list)}], Dice: {dice_mean:.4f}")

                per_subset_scores[subset_key] = (
                    float(np.mean(dice_all_wt)),
                    float(np.mean(dice_all_co)),
                    float(np.mean(dice_all_ec)),
                    float(np.mean(dice_all_mean)),
                )

        time_end = time.time()
        val_time = (time_end - time_start) / 60

        # Aggregate: average-of-averages across evaluated subsets
        dice_wt_mean = float(np.mean([v[0] for v in per_subset_scores.values()]))
        dice_co_mean = float(np.mean([v[1] for v in per_subset_scores.values()]))
        dice_ec_mean = float(np.mean([v[2] for v in per_subset_scores.values()]))
        dice_mean    = float(np.mean([v[3] for v in per_subset_scores.values()]))

        # Log summary
        if len(per_subset_scores) == 1:
            only_key, only_vals = next(iter(per_subset_scores.items()))
            logging.info(
                f"Epoch {epoch} validation time: {val_time:.2f} minutes\n"
                f"[{only_key}] Dice scores - WT: {only_vals[0]:.4f}, TC: {only_vals[1]:.4f}, "
                f"ET: {only_vals[2]:.4f}, Mean: {only_vals[3]:.4f}"
            )
        else:
            logging.info(
                f"Epoch {epoch} validation time: {val_time:.2f} minutes over {len(per_subset_scores)} subsets\n"
                f"Avg Dice - WT: {dice_wt_mean:.4f}, TC: {dice_co_mean:.4f}, "
                f"ET: {dice_ec_mean:.4f}, Mean: {dice_mean:.4f}"
            )
            for k, v in per_subset_scores.items():
                logging.info(f"  {k}: WT={v[0]:.4f}, TC={v[1]:.4f}, ET={v[2]:.4f}, Mean={v[3]:.4f}")

        # Tensorboard logging (aggregate only, keeps dashboard clean)
        self.writer.add_scalar("val/dice_wt", dice_wt_mean, epoch)
        self.writer.add_scalar("val/dice_co", dice_co_mean, epoch)
        self.writer.add_scalar("val/dice_ec", dice_ec_mean, epoch)
        self.writer.add_scalar("val/dice_mean", dice_mean, epoch)
        # Per-subset breakdown (useful when validating on multiple subsets in unified mode)
        if len(per_subset_scores) > 1:
            for k, v in per_subset_scores.items():
                self.writer.add_scalar(f"val_subset/{k}/wt", v[0], epoch)
                self.writer.add_scalar(f"val_subset/{k}/tc", v[1], epoch)
                self.writer.add_scalar(f"val_subset/{k}/et", v[2], epoch)
                self.writer.add_scalar(f"val_subset/{k}/mean", v[3], epoch)

        # Save best model
        if dice_mean > self.best_dice:
            self.best_dice = dice_mean
            self.best_epoch = epoch
            self.best_wt = dice_wt_mean
            self.best_co = dice_co_mean
            self.best_ec = dice_ec_mean

            self.save_checkpoint(epoch, is_best=True)
            logging.info(f"New best model saved with dice: {dice_mean:.4f}")

        return dice_mean
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_type": self.model_type,
            "student_modalities": self.student_modalities,
            # training_mode tells the evaluator whether this checkpoint was trained as
            # a per-subset specialist (in_channels = |subset|) or a unified model
            # (in_channels = 4, all 15 expert pools trained).
            "training_mode": self.training_mode,
            "student_state_dict": self.student_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "region_adaptor_state_dict": self.region_adaptor.state_dict(),
            "best_dice": self.best_dice,
            "best_epoch": self.best_epoch,
            "config": self.cfg
        }
        
        if is_best:
            save_path = os.path.join(self.save_model_path, "best_model.pth")
        else:
            save_path = os.path.join(self.save_model_path, f"checkpoint_{epoch}.pth")
        
        torch.save(checkpoint, save_path)
        logging.info(f"Checkpoint saved to {save_path}")
    
    def train(self):
        """Main training loop"""
        # Backward compatibility guard: make sure all new attributes exist
        self._ensure_runtime_defaults()
        logging.info("=" * 60)
        logging.info("Starting Training")
        logging.info("=" * 60)
        logging.info(f"Model: {self.model_type}, Student modalities: {self.student_modality_key}")
        
        train_start_time = time.time()
        
        for epoch in range(self.start_epoch, self.max_epoch):
            # Adjust learning rate with optional schedule and warmup
            if self.lr_schedule == "poly":
                current_lr = self.lr * (1.0 - float(epoch) / float(max(1, self.max_epoch))) ** 0.9
            elif self.lr_schedule == "cosine":
                import math
                current_lr = self.lr * 0.5 * (1.0 + math.cos(math.pi * float(epoch) / float(max(1, self.max_epoch))))
            else:  # const
                current_lr = self.lr
            if self.lr_warmup_epochs and epoch < self.lr_warmup_epochs:
                warmup_scale = float(epoch + 1) / float(self.lr_warmup_epochs)
                current_lr = max(self.lr * 0.1, current_lr * warmup_scale)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            logging.info(f"\nEpoch {epoch}/{self.max_epoch}, Learning rate: {current_lr:.6f}")
            self.writer.add_scalar("train/lr", current_lr, epoch)
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Save checkpoint periodically
            if epoch % 200 == 0:
                self.save_checkpoint(epoch)
            
            # Validate after configured start epoch and at configured interval
            if epoch >= self.val_start_epoch and ((epoch - self.val_start_epoch) % max(1, self.val_interval) == 0):
                self.validate(epoch)
            
            logging.info(f"Epoch {epoch} completed. Best dice so far: {self.best_dice:.4f}")
        
        # Training completed
        train_end_time = time.time()
        total_time = (train_end_time - train_start_time) / 3600
        
        self.writer.close()
        
        logging.info("=" * 50)
        logging.info("Training completed!")
        logging.info(f"Total training time: {total_time:.2f} hours")
        logging.info(f"Best epoch: {self.best_epoch}, Best dice: {self.best_dice:.4f}")
        logging.info(f"Best scores - WT: {self.best_wt:.4f}, TC: {self.best_co:.4f}, ET: {self.best_ec:.4f}")
        logging.info("=" * 50)
