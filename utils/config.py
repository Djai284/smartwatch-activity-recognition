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
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    n_splits: int = 5
    num_workers: int = 4
    early_stopping_patience: int = 7
    
    # Model parameters
    num_classes: int = None  # Set based on data
    num_channels: int = 6  # 3 acc + 3 gyro
    
    # Augmentation
    use_augmentation: bool = True
    jitter_strength: float = 0.1
    
    # Device
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __post_init__(self):
        """Create necessary directories"""
        self.model_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
    def save(self, path):
        """Save configuration to JSON"""
        import json
        config_dict = {k: str(v) if isinstance(v, Path) else v 
                      for k, v in self.__dict__.items()}
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)

