# config.py
from pathlib import Path
import torch
from dataclasses import dataclass
import os

@dataclass
class Config:
    """Configuration for training pipeline"""
    # Paths
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    log_dir: Path = Path("logs")
    
    # Data parameters
    window_length: int = 500
    target_sampling_rate: int = 50
    
    # Training parameters
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    n_splits: int = 5
    early_stopping_patience: int = 7
    # Memory management
    batch_size: int = 16  # Reduced batch size
    num_workers: int = 2  # Reduced worker count
    prefetch_factor: int = 2  # Control prefetching
    pin_memory: bool = True
    
    # Model parameters
    num_classes: int = None  # Set based on data
    num_channels: int = 6  # 3 acc + 3 gyro
    
    # Augmentation
    use_augmentation: bool = True
    jitter_strength: float = 0.1

    
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using device: {device}")
    # print if using cuda
    if device.type == 'cuda':
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    def __post_init__(self):
        """Create necessary directories"""
        self.model_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        # Set CUDA options for memory efficiency
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
        
    def save(self, path):
        """Save configuration to JSON"""
        import json
        config_dict = {k: str(v) if isinstance(v, Path) else v 
                      for k, v in self.__dict__.items()}
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)

