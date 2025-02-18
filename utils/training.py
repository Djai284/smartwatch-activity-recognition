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
from utils.utils import setup_logging, log_memory_usage, \
    log_model_summary, log_epoch_summary, log_batch_metrics, log_epoch_metrics, MemoryEfficientDataset, \
        custom_collate, check_batch_shapes, validate_data_shapes

import os

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer):
    """Train for one epoch with improved logging"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar
    progress_bar = tqdm(
        train_loader,
        desc=f'Epoch {epoch}',
        leave=True,
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]'
    )
    
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
        
        # Update progress bar every batch
        avg_loss = running_loss / (batch_idx + 1)
        acc = 100. * correct / total
        
        # Update progress bar description
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'acc': f'{acc:.2f}%'
        })
        
        # Log to tensorboard (less frequently)
        if batch_idx % 100 == 0:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Batch/Loss', avg_loss, step)
            writer.add_scalar('Batch/Accuracy', acc, step)
    
    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    """Validate with progress bar"""
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
            
            # Update progress bar
            val_loader.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
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
    """Train model with cross-validation using custom collate function"""
    logger = setup_logging(config)
    logger.info("Starting training pipeline")
    log_memory_usage(logger)
    
    # Create indices for cross-validation
    indices = np.arange(len(data))
    kfold = KFold(n_splits=config.n_splits, shuffle=True, random_state=42)
    cv_results = []
    
    # Get number of classes
    temp_encoder = LabelEncoder()
    temp_encoder.fit(data['activity_name'])
    config.num_classes = len(temp_encoder.classes_)
    logger.info(f"Number of classes: {config.num_classes}")
    
    # Log sequence length information
    sequence_lengths = [len(x[0]) for x in data['sig_array']]
    logger.info(f"Sequence length stats:")
    logger.info(f"Min length: {min(sequence_lengths)}")
    logger.info(f"Max length: {max(sequence_lengths)}")
    logger.info(f"Mean length: {np.mean(sequence_lengths):.2f}")
    logger.info(f"Target length: {config.window_length}")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        logger.info(f"\n{'='*20} Fold {fold + 1}/{config.n_splits} {'='*20}")
        log_memory_usage(logger)
        
        # Create datasets
        logger.info("Creating datasets")
        train_dataset = MemoryEfficientDataset(
            data,
            train_idx,
            preprocessor=SignalPreprocessor(config.target_sampling_rate),
            augmenter=TimeSeriesAugmentation(
                jitter_strength=config.jitter_strength,
                rotation_strength=config.jitter_strength
            ) if config.use_augmentation else None,
            target_length=config.window_length
        )
        
        val_dataset = MemoryEfficientDataset(
            data,
            val_idx,
            preprocessor=SignalPreprocessor(config.target_sampling_rate),
            target_length=config.window_length
        )
        
        logger.info(f"Train set size: {len(train_dataset)}")
        logger.info(f"Validation set size: {len(val_dataset)}")
        logger.info("Validating data shapes...")
        invalid_indices = validate_data_shapes(data, logger)
        if invalid_indices:
            logger.warning("Found invalid shapes in data. Consider cleaning the dataset.")
        
        # Create dataloaders with custom collate function
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=custom_collate
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=custom_collate
        )

        logger.info("Checking batch shapes...")
        input_shape, target_shape = check_batch_shapes(train_loader, logger)
        logger.info(f"Input shape: {input_shape}, Target shape: {target_shape}")
        
        # Initialize model
        model = model_class(
            num_classes=config.num_classes,
            window_length=config.window_length
        ).to(config.device)
        
        log_model_summary(logger, model)
        
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
        
        # Training loop
        # Training loop
        best_val_loss = float('inf')
        early_stopping_counter = 0
        
        for epoch in range(config.num_epochs):
            # Train
            progress_bar = tqdm(
                train_loader,
                desc=f'Epoch {epoch+1}/{config.num_epochs}',
                ncols=100,
                leave=True
            )
            
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                # Clear cache periodically
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                
                inputs = inputs.to(config.device, non_blocking=True)
                targets = targets.to(config.device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                avg_loss = train_loss / (batch_idx + 1)
                avg_acc = 100. * train_correct / train_total
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{avg_acc:.2f}%'
                })
            
            # Calculate final training metrics
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            # Validate with progress bar
            val_progress = tqdm(
                val_loader,
                desc='Validating',
                ncols=100,
                leave=False
            )
            val_loss, val_acc, val_preds, val_targets = validate(
                model, val_progress, criterion, config.device
            )
            
            # Log epoch summary
            logger.info(
                f"Epoch [{epoch+1}/{config.num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
            )
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                model_path = config.model_dir / f"best_model_fold_{fold}.pt"
                torch.save(model.state_dict(), model_path)
                logger.info(f"New best model saved (val_loss: {val_loss:.4f})")
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            # Clear memory
            torch.cuda.empty_cache()
        
        # Save fold results
        fold_results = {
            'fold': fold,
            'best_val_loss': float(best_val_loss),
            'final_val_acc': float(val_acc),
            'classification_report': classification_report(
                val_targets, val_preds,
                target_names=train_dataset.label_encoder.classes_,
                output_dict=True
            )
        }
        
        cv_results.append(fold_results)
        logger.info("\nFold Memory Usage:")
        log_memory_usage(logger)
        
        # Clear memory between folds
        del model, train_dataset, val_dataset, train_loader, val_loader
        torch.cuda.empty_cache()
    
    return cv_results

def optimize_for_mobile(model, config):
    """Optimize model for mobile deployment"""
    print("Optimizing model for mobile deployment...")
    
    # Move model to CPU for quantization
    model = model.cpu()
    
    # Quantize model
    print("Quantizing model...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear, nn.Conv1d}, 
        dtype=torch.qint8
    )
    
    # Create example input (on CPU)
    example_input = torch.randn(
        1, config.num_channels, config.window_length,
        device='cpu'  # Ensure input is on CPU
    )
    
    # Export to TorchScript
    print("Tracing model...")
    traced_model = torch.jit.trace(quantized_model, example_input)
    
    # Save optimized model
    output_path = config.model_dir / "mobile_optimized_model.pt"
    traced_model.save(str(output_path))
    
    # Calculate and log model sizes
    original_size = os.path.getsize(config.model_dir / "best_model_fold_0.pt")
    optimized_size = os.path.getsize(output_path)
    
    print(f"Original model size: {original_size/1024:.2f} KB")
    print(f"Optimized model size: {optimized_size/1024:.2f} KB")
    print(f"Size reduction: {100*(1-optimized_size/original_size):.1f}%")
    
    return quantized_model