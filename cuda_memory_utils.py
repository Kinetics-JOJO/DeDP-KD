"""
CUDA Memory Management Utilities for Deep Learning Training
Helps prevent cuDNN algorithm errors and out-of-memory issues
"""

import torch
import gc
import os


def setup_cuda_memory_optimization():
    """
    Setup CUDA memory optimizations to prevent cuDNN errors
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        return
    
    # Environment variables for memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Set memory fraction (use 90% of available GPU memory)
    torch.cuda.set_per_process_memory_fraction(0.9)
    
    print("CUDA memory optimizations applied.")


def enable_mixed_precision():
    """
    Enable automatic mixed precision training for memory efficiency
    """
    try:
        from torch.cuda.amp import autocast, GradScaler
        print("Mixed precision training enabled.")
        return GradScaler()
    except ImportError:
        print("Mixed precision not available in this PyTorch version.")
        return None


def print_gpu_memory_usage():
    """
    Print current GPU memory usage
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"GPU Memory Usage:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Total:     {total:.2f} GB")
        print(f"  Free:      {total - reserved:.2f} GB")


def clear_gpu_cache():
    """
    Aggressively clear GPU cache
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


class CUDAMemoryManager:
    """
    Context manager for CUDA memory optimization during training
    """
    def __init__(self, clear_cache_every_n_steps=10):
        self.clear_cache_every_n_steps = clear_cache_every_n_steps
        self.step_count = 0
    
    def step(self):
        """Call this after each training step"""
        self.step_count += 1
        if self.step_count % self.clear_cache_every_n_steps == 0:
            clear_gpu_cache()
    
    def __enter__(self):
        setup_cuda_memory_optimization()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        clear_gpu_cache()