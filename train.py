import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch

from utils.config import Config
from model import IMUCNN, TimeSeriesAugmentation
from utils.data_processing import SignalPreprocessor, ExerciseDataset
from utils.training import train_model_with_cv, optimize_for_mobile

# Load configuration
config = Config()

# Load data
print("Loading data...")
data = pd.read_pickle(config.data_dir / 'model_data_window_500_space_50.pkl')

# Set number of classes based on data
config.num_classes = len(data['activity_name'].unique())
print(f"Number of classes: {config.num_classes}")

# Add sampling rate information to dataset
data['sampling_rate'] = data['dataset'].map({
    'mmfit': 100,
    'har_data': 100,
    'reco': 50
})

# Train with cross-validation
print("Starting cross-validation training...")
cv_results = train_model_with_cv(IMUCNN, data, config)

# Get best model
best_fold = min(cv_results, key=lambda x: x['best_val_loss'])['fold']
print(f"\nBest model was from fold {best_fold}")
print(f"Best validation loss: {cv_results[best_fold]['best_val_loss']:.4f}")
print(f"Best validation accuracy: {cv_results[best_fold]['final_val_acc']:.2f}%")

# Load best model
best_model = IMUCNN(
    num_classes=config.num_classes,
    window_length=config.window_length
).to(config.device)

best_model.load_state_dict(torch.load(
    config.model_dir / f"best_model_fold_{best_fold}.pt"
))

# Optimize for mobile
print("\nOptimizing model for mobile deployment...")
mobile_model = optimize_for_mobile(best_model, config)
print("Mobile optimization complete!")