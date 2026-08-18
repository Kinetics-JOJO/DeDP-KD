"""
Prompt modules for prompt-based knowledge distillation
Inspired by DualPrompt for continual learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import itertools
import numpy as np


class ModalityCombination:
    """Generate all possible modality combinations"""
    
    MODALITIES = ['T1', 'T2', 'T1ce', 'FLAIR']
    MODALITY_INDICES = {'T1': 0, 'T2': 1, 'T1ce': 2, 'FLAIR': 3}
    
    @classmethod
    def get_all_combinations(cls):
        """Get all possible non-empty combinations of modalities"""
        combinations = []
        for r in range(1, len(cls.MODALITIES) + 1):
            for combo in itertools.combinations(cls.MODALITIES, r):
                combinations.append(combo)
        return combinations
    
    @classmethod
    def get_combination_key(cls, modality_indices: List[int]) -> str:
        """Convert modality indices to combination key"""
        modalities = [cls.MODALITIES[i] for i in sorted(modality_indices)]
        return '+'.join(modalities)
    
    @classmethod
    def get_indices_from_key(cls, key: str) -> List[int]:
        """Convert combination key to modality indices"""
        modalities = key.split('+')
        return [cls.MODALITY_INDICES[m] for m in modalities]


class PromptPool(nn.Module):
    """Pool of learnable prompts"""
    
    def __init__(
        self,
        num_prompts: int,
        prompt_length: int,
        embedding_dim: int,
        prompt_init: str = 'uniform',
        prompt_key_init: str = 'uniform'
    ):
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_length = prompt_length
        self.embedding_dim = embedding_dim
        
        # Initialize prompt embeddings
        self.prompts = nn.Parameter(
            torch.zeros(num_prompts, prompt_length, embedding_dim)
        )
        
        # Initialize prompt keys for selection
        self.prompt_keys = nn.Parameter(
            torch.zeros(num_prompts, embedding_dim)
        )
        
        # Initialize parameters
        self._init_prompts(prompt_init)
        self._init_keys(prompt_key_init)
        
    def _init_prompts(self, init_type):
        """Initialize prompt embeddings"""
        if init_type == 'uniform':
            nn.init.uniform_(self.prompts, -0.02, 0.02)
        elif init_type == 'normal':
            nn.init.normal_(self.prompts, std=0.02)
        elif init_type == 'xavier':
            nn.init.xavier_uniform_(self.prompts)
            
    def _init_keys(self, init_type):
        """Initialize prompt keys"""
        if init_type == 'uniform':
            nn.init.uniform_(self.prompt_keys, -1, 1)
        elif init_type == 'normal':
            nn.init.normal_(self.prompt_keys, std=1.0)
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(self.prompt_keys)
            
    def forward(self, query_features: torch.Tensor, top_k: int = 1) -> torch.Tensor:
        """
        Select prompts based on query features
        
        Args:
            query_features: Features to match against keys [B, D]
            top_k: Number of prompts to select
            
        Returns:
            Selected prompts [B, top_k * prompt_length, D]
        """
        B = query_features.shape[0]
        
        # Compute similarity between query and keys
        query_norm = F.normalize(query_features, dim=-1)  # [B, D]
        key_norm = F.normalize(self.prompt_keys, dim=-1)  # [N, D]
        
        similarity = torch.matmul(query_norm, key_norm.T)  # [B, N]
        
        # Select top-k prompts
        _, indices = torch.topk(similarity, top_k, dim=-1)  # [B, top_k]
        
        # Gather selected prompts
        selected_prompts = []
        for b in range(B):
            batch_prompts = []
            for k in range(top_k):
                idx = indices[b, k]
                batch_prompts.append(self.prompts[idx])
            selected_prompts.append(torch.cat(batch_prompts, dim=0))
            
        return torch.stack(selected_prompts)  # [B, top_k * prompt_length, D]


class DualPromptModule(nn.Module):
    """
    Dual Prompt module with general and expert prompts
    General prompts: shared across all modality combinations (early layers)
    Expert prompts: specific to modality combinations (later layers)
    """
    
    def __init__(
        self,
        num_layers: int = 12,
        embedding_dim: int = 768,
        general_prompt_length: int = 5,
        expert_prompt_length: int = 5,
        general_layers: List[int] = None,
        expert_layers: List[int] = None
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Define which layers get which prompts
        if general_layers is None:
            # First half layers get general prompts
            self.general_layers = list(range(num_layers // 2))
        else:
            self.general_layers = general_layers
            
        if expert_layers is None:
            # Second half layers get expert prompts
            self.expert_layers = list(range(num_layers // 2, num_layers))
        else:
            self.expert_layers = expert_layers
        
        # Create general prompt pool (shared across all modalities)
        self.general_prompt_pool = PromptPool(
            num_prompts=1,  # Single shared general prompt
            prompt_length=general_prompt_length,
            embedding_dim=embedding_dim
        )
        
        # Create expert prompt pools for each modality combination
        self.expert_prompt_pools = nn.ModuleDict()
        combinations = ModalityCombination.get_all_combinations()
        
        for combo in combinations:
            key = '+'.join(combo)
            self.expert_prompt_pools[key] = PromptPool(
                num_prompts=1,  # One expert prompt per combination
                prompt_length=expert_prompt_length,
                embedding_dim=embedding_dim
            )
        
        self.general_prompt_length = general_prompt_length
        self.expert_prompt_length = expert_prompt_length
        
    def get_prompts_for_layer(
        self,
        layer_idx: int,
        modality_key: str,
        query_features: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Get prompts for a specific layer and modality combination
        
        Args:
            layer_idx: Index of the transformer layer
            modality_key: Key representing modality combination (e.g., 'T1+T2')
            query_features: Features for prompt selection [B, D]
            
        Returns:
            Prompts for this layer [B, prompt_length, D] or None
        """
        B = query_features.shape[0] if query_features is not None else 1
        
        if layer_idx in self.general_layers:
            # Return general prompts
            if query_features is not None:
                prompts = self.general_prompt_pool(query_features, top_k=1)
            else:
                # Return the single general prompt repeated for batch
                prompts = self.general_prompt_pool.prompts[0].unsqueeze(0).repeat(B, 1, 1)
            return prompts
            
        elif layer_idx in self.expert_layers:
            # Return expert prompts for this modality combination
            if modality_key in self.expert_prompt_pools:
                if query_features is not None:
                    prompts = self.expert_prompt_pools[modality_key](query_features, top_k=1)
                else:
                    # Return the expert prompt for this combination
                    prompts = self.expert_prompt_pools[modality_key].prompts[0].unsqueeze(0).repeat(B, 1, 1)
                return prompts
            else:
                # Fallback to general prompt if combination not found
                if query_features is not None:
                    prompts = self.general_prompt_pool(query_features, top_k=1)
                else:
                    prompts = self.general_prompt_pool.prompts[0].unsqueeze(0).repeat(B, 1, 1)
                return prompts
        
        return None
    
    def forward(
        self,
        features: torch.Tensor,
        layer_idx: int,
        modality_indices: List[int]
    ) -> torch.Tensor:
        """
        Add prompts to features at a specific layer
        
        Args:
            features: Input features [B, N, D]
            layer_idx: Current layer index
            modality_indices: Indices of active modalities
            
        Returns:
            Features with prompts prepended [B, N + prompt_length, D]
        """
        modality_key = ModalityCombination.get_combination_key(modality_indices)
        
        # Get prompts for this layer
        prompts = self.get_prompts_for_layer(
            layer_idx,
            modality_key,
            query_features=features.mean(dim=1)  # Use mean pooled features as query
        )
        
        if prompts is not None:
            # Prepend prompts to features
            features = torch.cat([prompts, features], dim=1)
        
        return features


class SwinStagePromptModule(nn.Module):
    """
    Stage-aware prompt module for Swin-UNETR.

    Swin uses multi-stage token dims (e.g., 48/96/192/384), so a single prompt
    embedding dimension will mismatch most stages. This module maintains
    separate prompt pools per stage to align dimensions.
    """

    def __init__(
        self,
        stage_dims: List[int],
        general_prompt_length: int = 5,
        expert_prompt_length: int = 5,
        general_stages: Optional[List[int]] = None,
        expert_stages: Optional[List[int]] = None
    ):
        super().__init__()
        self.stage_dims = list(stage_dims)
        self.num_stages = len(self.stage_dims)

        if general_stages is None:
            self.general_stages = list(range(self.num_stages // 2))
        else:
            self.general_stages = general_stages

        if expert_stages is None:
            self.expert_stages = list(range(self.num_stages // 2, self.num_stages))
        else:
            self.expert_stages = expert_stages

        # General prompt pool per stage (shared across modalities within a stage)
        self.general_prompt_pools = nn.ModuleList([
            PromptPool(
                num_prompts=1,
                prompt_length=general_prompt_length,
                embedding_dim=dim
            )
            for dim in self.stage_dims
        ])

        # Expert prompt pools per stage and modality combination
        self.expert_prompt_pools = nn.ModuleList()
        combinations = ModalityCombination.get_all_combinations()
        for dim in self.stage_dims:
            pools = nn.ModuleDict()
            for combo in combinations:
                key = '+'.join(combo)
                pools[key] = PromptPool(
                    num_prompts=1,
                    prompt_length=expert_prompt_length,
                    embedding_dim=dim
                )
            self.expert_prompt_pools.append(pools)

        self.general_prompt_length = general_prompt_length
        self.expert_prompt_length = expert_prompt_length

    def set_active_expert_key(self, active_key: str):
        """Freeze expert pools not matching the active modality key."""
        for stage_pools in self.expert_prompt_pools:
            for key, pool in stage_pools.items():
                requires = (key == active_key)
                for param in pool.parameters():
                    param.requires_grad = requires

    def get_prompts_for_stage(
        self,
        stage_idx: int,
        modality_key: str,
        query_features: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Get prompts for a specific Swin stage."""
        if stage_idx < 0 or stage_idx >= self.num_stages:
            return None

        B = query_features.shape[0] if query_features is not None else 1
        use_expert = stage_idx in self.expert_stages

        if use_expert:
            pools = self.expert_prompt_pools[stage_idx]
            if modality_key in pools:
                pool = pools[modality_key]
            else:
                pool = self.general_prompt_pools[stage_idx]
        else:
            pool = self.general_prompt_pools[stage_idx]

        if query_features is not None:
            return pool(query_features, top_k=1)

        return pool.prompts[0].unsqueeze(0).repeat(B, 1, 1)


class PromptedTransformerWrapper(nn.Module):
    """
    Wrapper to add prompts to transformer-based models
    Works with UNETR and Swin UNETR
    """
    
    def __init__(
        self,
        model: nn.Module,
        prompt_module: DualPromptModule,
        model_type: str = 'unetr'  # 'unetr' or 'swin_unetr'
    ):
        super().__init__()
        self.model = model
        self.prompt_module = prompt_module
        self.model_type = model_type
        
        # Register minimal, safe prompt injection hooks
        if self.model_type == 'unetr':
            self._register_unetr_hooks()
        elif self.model_type == 'swin_unetr':
            self._register_swin_hooks()
        
    def forward(self, x: torch.Tensor, modality_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with prompt injection
        
        Args:
            x: Input tensor [B, C, H, W, D]
            modality_indices: Indices of active modalities
            
        Returns:
            features: Encoder features
            logits: Segmentation output
        """
        if self.model_type == 'unetr':
            return self._forward_unetr(x, modality_indices)
        elif self.model_type == 'swin_unetr':
            return self._forward_swin_unetr(x, modality_indices)
        else:
            # Fallback to standard forward
            return self.model(x)
    
    def _forward_unetr(self, x: torch.Tensor, modality_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for UNETR with prompt injection (hooks already installed)."""
        # Expose modality indices for hooks during this forward
        self._current_modality_indices = modality_indices
        try:
            return self.model(x)
        finally:
            # Clean up to avoid leaking context across calls
            if hasattr(self, '_current_modality_indices'):
                delattr(self, '_current_modality_indices')
    
    def _forward_swin_unetr(self, x: torch.Tensor, modality_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for Swin UNETR with prompt injection (lightweight hooks)."""
        # Expose modality indices for hooks during this forward
        self._current_modality_indices = modality_indices
        try:
            return self.model(x)
        finally:
            # Clean up to avoid leaking context across calls
            if hasattr(self, '_current_modality_indices'):
                delattr(self, '_current_modality_indices')

    def _register_unetr_hooks(self):
        """Register per-layer additive prompt bias hooks on ViT blocks for UNETR.
        This is a safe injection that preserves token length and shapes.
        """
        try:
            # Underlying MONAI UNETR lives at self.model.model
            vit = getattr(self.model, 'model', None)
            if vit is None or not hasattr(vit, 'vit'):
                return
            vit_core = vit.vit
            if not hasattr(vit_core, 'blocks'):
                return
            blocks = vit_core.blocks
        except Exception:
            return

        layer_counter = {'idx': 0}

        def make_pre_hook(layer_idx: int):
            def pre_hook(module, inputs):
                # inputs is a tuple (x, ...), we only modify x if shape compatible
                if not inputs:
                    return inputs
                x = inputs[0]
                if not isinstance(x, torch.Tensor) or x.dim() != 3:
                    return inputs  # Expect [B, N, D]
                try:
                    B, N, D = x.shape
                    # Query features for prompt selection
                    query = x.mean(dim=1)  # [B, D]
                    modality_key_indices = []  # not used here; pass through indices via closure
                    # Use stored modality indices if available via attribute set in forward
                    modality_indices = getattr(self, '_current_modality_indices', None)
                    if modality_indices is None:
                        modality_indices = []
                    prompts = self.prompt_module.get_prompts_for_layer(
                        layer_idx=layer_idx,
                        modality_key=ModalityCombination.get_combination_key(modality_indices) if modality_indices else 'T1',
                        query_features=query
                    )
                    # Convert prompts to additive bias [B, D]
                    if prompts is not None:
                        prompt_bias = prompts.mean(dim=1)  # [B, D]
                        x = x + prompt_bias.unsqueeze(1)  # broadcast across tokens
                        return (x, *inputs[1:])
                except Exception:
                    return inputs
                return inputs
            return pre_hook

        # Attach hooks to each transformer block sequentially
        for idx, blk in enumerate(blocks):
            try:
                blk.register_forward_pre_hook(make_pre_hook(idx))
            except Exception:
                continue

    def _register_swin_hooks(self):
        """Register lightweight additive prompt bias on SwinViT input and early blocks.
        This avoids changing sequence/window shapes. Per-layer mapping is best-effort.
        """
        try:
            swin = getattr(self.model, 'model', None)
            if swin is None or not hasattr(swin, 'swinViT'):
                return
            swin_core = swin.swinViT
        except Exception:
            return

        def get_modality_key():
            modality_indices = getattr(self, '_current_modality_indices', None)
            if modality_indices is None:
                modality_indices = []
            return ModalityCombination.get_combination_key(modality_indices) if modality_indices else 'T1'

        def make_stage_pre(stage_idx: int):
            def pre_hook(module, inputs):
                if not inputs:
                    return inputs
                x = inputs[0]
                if not isinstance(x, torch.Tensor) or x.dim() != 3:
                    return inputs
                try:
                    query = x.mean(dim=1)
                    modality_key = get_modality_key()
                    if hasattr(self.prompt_module, 'get_prompts_for_stage'):
                        prompts = self.prompt_module.get_prompts_for_stage(
                            stage_idx=stage_idx,
                            modality_key=modality_key,
                            query_features=query
                        )
                    else:
                        prompts = self.prompt_module.get_prompts_for_layer(
                            layer_idx=stage_idx,
                            modality_key=modality_key,
                            query_features=query
                        )
                    if prompts is None or prompts.shape[-1] != x.shape[-1]:
                        return inputs
                    prompt_bias = prompts.mean(dim=1)
                    x = x + prompt_bias.unsqueeze(1)
                    return (x, *inputs[1:])
                except Exception:
                    return inputs
                return inputs
            return pre_hook

        # Prefer stage-level hooks if available (aligns with Swin multi-stage dims)
        stages = None
        if hasattr(swin_core, 'layers'):
            try:
                stages = list(swin_core.layers)
            except Exception:
                stages = None

        if stages:
            for stage_idx, stage in enumerate(stages):
                try:
                    stage.register_forward_pre_hook(make_stage_pre(stage_idx))
                except Exception:
                    continue
            return

        # Fallback: attach to modules that likely represent transformer blocks
        layer_idx_counter = 0

        def make_generic_pre(layer_idx: int):
            def pre_hook(module, inputs):
                if not inputs:
                    return inputs
                x = inputs[0]
                if not isinstance(x, torch.Tensor) or x.dim() != 3:
                    return inputs
                try:
                    query = x.mean(dim=1)
                    modality_key = get_modality_key()
                    if hasattr(self.prompt_module, 'get_prompts_for_stage'):
                        stage_idx = layer_idx % max(1, getattr(self.prompt_module, 'num_stages', 1))
                        prompts = self.prompt_module.get_prompts_for_stage(
                            stage_idx=stage_idx,
                            modality_key=modality_key,
                            query_features=query
                        )
                    else:
                        prompts = self.prompt_module.get_prompts_for_layer(
                            layer_idx=layer_idx,
                            modality_key=modality_key,
                            query_features=query
                        )
                    if prompts is not None and prompts.shape[-1] == x.shape[-1]:
                        prompt_bias = prompts.mean(dim=1)
                        x = x + prompt_bias.unsqueeze(1)
                        return (x, *inputs[1:])
                except Exception:
                    return inputs
                return inputs
            return pre_hook

        for name, submodule in swin_core.named_modules():
            if 'block' in name.lower() or 'blocks' in name.lower():
                try:
                    submodule.register_forward_pre_hook(make_generic_pre(layer_idx_counter))
                    layer_idx_counter += 1
                except Exception:
                    continue
    
    def freeze_encoder(self):
        """Freeze encoder parameters"""
        if hasattr(self.model, 'freeze_encoder'):
            self.model.freeze_encoder()
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters"""
        if hasattr(self.model, 'unfreeze_encoder'):
            self.model.unfreeze_encoder()


class RegionAdaptorHeads(nn.Module):
    """
    Lightweight region expert heads operating on final logits (N, C, D, H, W).
    Each region has a learnable prompt embedding that is mapped to FiLM (gamma/beta)
    to modulate the shared logits, followed by a 1x1x1 conv to produce a binary mask logit.

    This avoids invasive edits to backbone/decoder while enabling per-region specialization.
    """

    def __init__(
        self,
        in_channels: int = 4,
        embedding_dim: int = 128,
        num_regions: int = 3,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_regions = num_regions

        # Learnable region prompts: 0->WT, 1->TC, 2->ET
        self.region_prompts = nn.Parameter(torch.zeros(num_regions, embedding_dim))
        nn.init.normal_(self.region_prompts, std=0.02)

        # Map prompt -> FiLM params (gamma, beta) for channel-wise modulation of logits
        self.film_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channels * 2)
        )

        # Separate 1x1 conv head per region
        self.region_heads = nn.ModuleList([
            nn.Conv3d(in_channels, 1, kernel_size=1, bias=True) for _ in range(num_regions)
        ])

    def forward(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            logits: [N, C, D, H, W] final logits from the student model

        Returns:
            logits_wt, logits_tc, logits_et: each [N, 1, D, H, W]
        """
        N, C = logits.shape[:2]
        outputs: List[torch.Tensor] = []
        for r in range(self.num_regions):
            prompt = self.region_prompts[r]  # [E]
            film = self.film_mlp(prompt)  # [2C]
            gamma, beta = torch.split(film, C)
            # Bound FiLM to prevent exploding modulation
            gamma = torch.tanh(gamma)  # in [-1,1]
            beta = torch.clamp(beta, min=-5.0, max=5.0)
            # Reshape to [1, C, 1, 1, 1] for broadcasting
            gamma = gamma.view(1, C, 1, 1, 1)
            beta = beta.view(1, C, 1, 1, 1)
            modulated = logits * (1 + gamma) + beta
            out = self.region_heads[r](modulated)
            outputs.append(out)

        # Ensure exactly three outputs are returned
        if self.num_regions >= 3:
            return outputs[0], outputs[1], outputs[2]
        elif self.num_regions == 2:
            # If only two regions are defined, pad ET as zeros for API consistency
            zero = torch.zeros_like(outputs[0])
            return outputs[0], outputs[1], zero
        else:
            # Single region defined; pad others
            zero = torch.zeros_like(outputs[0])
            return outputs[0], zero, zero