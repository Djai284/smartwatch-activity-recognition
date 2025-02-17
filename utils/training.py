import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
import time
from sklearn.preprocessing import LabelEncoder
from utils.data_processing import ExerciseDataset, SignalPreprocessor
from model import IMUCNN, TimeSeriesAugmentation

import os

def train_epoch(model, train_loader, criterion, optimizer, device, epoch_idx, writer):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch_idx}')
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss/(batch_idx+1),
            'acc': 100.*correct/total
        })
        
        # Log batch metrics
        step = epoch_idx * len(train_loader) + batch_idx
        writer.add_scalar('Batch/Loss', loss.item(), step)
        writer.add_scalar('Batch/Accuracy', 100.*correct/total, step)
    
    return running_loss/len(train_loader), 100.*correct/total

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    return running_loss/len(val_loader), 100.*correct/total, all_preds, all_targets

def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def train_model_with_cv(model_class, data, config):
    """Train model with cross-validation"""
    kfold = KFold(n_splits=config.n_splits, shuffle=True, random_state=42)
    cv_results = []
    
    # Prepare data for CV
    X = np.stack(data['sig_array'].values).astype(np.float32)  # Convert to numpy array
    y = data['activity_name'].values
    
    # Get sampling rates
    sampling_rates = data['dataset'].map({
        'mmfit': 100, 'har_data': 100, 'reco': 50
    }).values
    
    # Fit label encoder once for all folds
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    config.num_classes = len(label_encoder.classes_)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\nTraining Fold {fold + 1}/{config.n_splits}")
        
        # Initialize model
        model = model_class(
            num_classes=config.num_classes,
            window_length=config.window_length
        ).to(config.device)
        
        # Create datasets for this fold
        preprocessor = SignalPreprocessor(target_sampling_rate=config.target_sampling_rate)
        augmenter = TimeSeriesAugmentation(
            jitter_strength=config.jitter_strength,
            rotation_strength=config.jitter_strength
        ) if config.use_augmentation else None
        
        train_dataset = ExerciseDataset(
            X[train_idx], y[train_idx], sampling_rates[train_idx],
            preprocessor=preprocessor,
            augmenter=augmenter
        )
        
        val_dataset = ExerciseDataset(
            X[val_idx], y[val_idx], sampling_rates[val_idx],
            preprocessor=preprocessor
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        
        # Training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Initialize tensorboard writer
        writer = SummaryWriter(f"{config.log_dir}/fold_{fold}")
        
        # Training loop
        best_val_loss = float('inf')
        early_stopping_counter = 0
        
        for epoch in range(config.num_epochs):
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, 
                config.device, epoch, writer
            )
            
            # Validate
            val_loss, val_acc, val_preds, val_targets = validate(
                model, val_loader, criterion, config.device
            )
            
            # Log metrics
            writer.add_scalar(f'Loss/train', train_loss, epoch)
            writer.add_scalar(f'Loss/val', val_loss, epoch)
            writer.add_scalar(f'Accuracy/train', train_acc, epoch)
            writer.add_scalar(f'Accuracy/val', val_acc, epoch)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                torch.save(
                    model.state_dict(),
                    config.model_dir / f"best_model_fold_{fold}.pt"
                )
            else:
                early_stopping_counter += 1
            
            # Early stopping
            if early_stopping_counter >= config.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Save fold results
        fold_results = {
            'fold': fold,
            'best_val_loss': float(best_val_loss),
            'final_val_acc': float(val_acc),
            'classification_report': classification_report(
                val_targets, val_preds,
                target_names=label_encoder.classes_,
                output_dict=True
            )
        }
        
        cv_results.append(fold_results)
        writer.close()
    
    return cv_results

def optimize_for_mobile(model, config):
    """Optimize model for mobile deployment"""
    print("Optimizing model for mobile deployment...")
    
    # Quantize model
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv1d}, dtype=torch.qint8
    )
    
    # Create example input
    example_input = torch.randn(
        1, config.num_channels, config.window_length,
        device=config.device
    )
    
    # Export to TorchScript
    print("Tracing model...")
    traced_model = torch.jit.trace(quantized_model, example_input)
    
    # Save optimized model
    output_path = config.model_dir / "mobile_optimized_model.pt"
    traced_model.save(str(output_path))
    print(f"Optimized model saved to {output_path}")
    
    # Calculate and log model sizes
    original_size = os.path.getsize(config.model_dir / "best_model_fold_0.pt")
    optimized_size = os.path.getsize(output_path)
    print(f"Original model size: {original_size/1024:.2f} KB")
    print(f"Optimized model size: {optimized_size/1024:.2f} KB")
    print(f"Size reduction: {100*(1-optimized_size/original_size):.1f}%")
    
    return quantized_model