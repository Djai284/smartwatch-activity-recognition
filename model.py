import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# training.py
import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import json
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import optuna
from optuna.trial import Trial

def train_model_with_tensorboard(model, train_loader, val_loader, device, 
                               num_epochs=50, 
                               learning_rate=0.001, 
                               weight_decay=1e-5,
                               log_dir='runs/exercise_cnn',
                               save_dir='runs',
                               early_stopping_patience=7,
                               augmenter=None):
    """
    Train the model with TensorBoard logging
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to run the model on (cuda/cpu)
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
        log_dir: Directory for TensorBoard logs
        save_dir: Directory to save model checkpoints
        early_stopping_patience: Number of epochs to wait before early stopping
        augmenter: Optional data augmentation object
    
    Returns:
        Trained model and training history
    """
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Initialize variables for tracking metrics
    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Get a batch of training data for TensorBoard graph visualization
    dataiter = iter(train_loader)
    example_inputs, _ = next(dataiter)
    example_inputs = example_inputs.to(device)
    writer.add_graph(model, example_inputs)
    
    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Training phase
        model.train()
        train_losses = []
        train_preds = []
        train_true = []
        
        for X_batch, y_batch in train_loader:
            # Apply data augmentation if provided
            if augmenter is not None:
                X_batch = augmenter.apply(X_batch)
            
            # Move to device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_true.extend(y_batch.cpu().numpy())
        
        # Calculate training metrics
        train_loss = np.mean(train_losses)
        train_acc = accuracy_score(train_true, train_preds)
        
        # Validation phase
        model.eval()
        val_losses = []
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_losses.append(loss.item())
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(y_batch.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = np.mean(val_losses)
        val_acc = accuracy_score(val_true, val_preds)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Add model weights histograms
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print metrics
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the best model
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            print(f'New best model saved with val_loss: {val_loss:.4f}')
        else:
            epochs_without_improvement += 1
        
        # Save the model if it has the best accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_acc_model.pt'))
            print(f'New best accuracy model saved with val_acc: {val_acc:.4f}')
        
        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
        
        print('-' * 60)
    
    # Calculate and log final training time
    training_time = time.time() - start_time
    print(f'Total training time: {training_time:.2f} seconds')
    writer.add_text('Training_Summary', f'Training completed in {training_time:.2f} seconds')
    
    # Close TensorBoard writer
    writer.close()
    
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt')))
    
    return model, history


class IMUCNN(nn.Module):
    def __init__(self, num_classes=5, window_length=100):
        super().__init__()
        # Convolution layers
        self.conv1 = nn.Conv1d(
            in_channels=6,   
            out_channels=32, 
            kernel_size=9,   
            stride=1,
            padding=4        
        )
        self.bn1 = nn.BatchNorm1d(32)  
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=9,
            stride=1,
            padding=4        
        )
        self.bn2 = nn.BatchNorm1d(64)  
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=9,
            stride=1,
            padding=4        
        )
        self.bn3 = nn.BatchNorm1d(128)  
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=9,
            stride=1,
            padding=4        
        )
        self.bn4 = nn.BatchNorm1d(256)  
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        # Calculate output size after convolutions and pooling
        # With 'same' padding, only pooling reduces the size
        out_size = window_length
        out_size = out_size // 2  # after pool1
        out_size = out_size // 2  # after pool2
        out_size = out_size // 2  # after pool3
        out_size = out_size // 2
        
        # The final feature dimension
        self.fc_input_dim = 256 * out_size

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, 500)
        self.bn_fc1 = nn.BatchNorm1d(500)  
        self.dropout1 = nn.Dropout(p=0.5)  
        
        self.fc2 = nn.Linear(500, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)   
        self.dropout2 = nn.Dropout(p=0.5) 

        
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        x shape: [batch_size, 6, window_length]
        """
        # 1) First convolution + BatchNorm + ReLU + max pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # 2) Second convolution + BatchNorm + ReLU + max pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # 3) Third convolution + BatchNorm + ReLU + max pool
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # 3) Third convolution + BatchNorm + ReLU + max pool
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)

        # 4) Flatten the feature maps
        x = x.view(-1, self.fc_input_dim)

        # 5) Dense layers with BatchNorm
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)  

        return x

# Data augmentation for time series
class TimeSeriesAugmentation:
    def __init__(self, jitter_strength=0.1, rotation_strength=0.1):
        self.jitter_strength = jitter_strength
        self.rotation_strength = rotation_strength
    
    def add_noise(self, x):
        # Add random noise
        noise = torch.randn_like(x) * self.jitter_strength
        return x + noise
    
    def time_shift(self, x, max_shift=20):
        # Randomly shift the time series
        batch_size, channels, seq_len = x.shape
        shifted_x = torch.zeros_like(x)
        
        for i in range(batch_size):
            shift = torch.randint(-max_shift, max_shift+1, (1,)).item()
            if shift > 0:
                shifted_x[i, :, shift:] = x[i, :, :-shift]
                shifted_x[i, :, :shift] = x[i, :, :shift]
            elif shift < 0:
                shift = abs(shift)
                shifted_x[i, :, :-shift] = x[i, :, shift:]
                shifted_x[i, :, -shift:] = x[i, :, -shift:]
            else:
                shifted_x[i] = x[i]
                
        return shifted_x
    
    def scale(self, x, scale_range=(0.8, 1.2)):
        # Randomly scale the time series
        scale = torch.rand(1) * (scale_range[1] - scale_range[0]) + scale_range[0]
        return x * scale
    
    def apply(self, x):
        x = self.add_noise(x)
        x = self.time_shift(x)
        x = self.scale(x)
        return x
    

class FlexibleCNN(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        window_length,
        conv_layers=4,
        kernel_size=9,
        hidden_layers=[256, 128, 64],
        dropout_rate=0.5,
        activation_fn=F.relu,
        initial_filters=32
    ):
        super().__init__()
        self.activation_fn = activation_fn
        
        # Initialize lists to store layers
        self.conv_blocks = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        # Create convolutional layers dynamically
        in_channels = input_channels
        out_channels = initial_filters
        
        for _ in range(conv_layers):
            conv_block = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2  # Same padding
            )
            self.conv_blocks.append(conv_block)
            self.bn_layers.append(nn.BatchNorm1d(out_channels))
            self.pool_layers.append(nn.MaxPool1d(kernel_size=2))
            
            in_channels = out_channels
            out_channels *= 2  # Double the filters for each layer
        
        # Calculate the output size after convolutions and pooling
        out_size = window_length
        for _ in range(conv_layers):
            out_size = out_size // 2  # Effect of MaxPool
            
        # Calculate input dimension for the first fully connected layer
        self.fc_input_dim = in_channels * out_size
        
        # Create fully connected layers dynamically
        self.fc_layers = nn.ModuleList()
        self.fc_bn_layers = nn.ModuleList()
        self.fc_dropouts = nn.ModuleList()
        
        current_dim = self.fc_input_dim
        
        # Add hidden layers
        for hidden_dim in hidden_layers:
            self.fc_layers.append(nn.Linear(current_dim, hidden_dim))
            self.fc_bn_layers.append(nn.BatchNorm1d(hidden_dim))
            self.fc_dropouts.append(nn.Dropout(p=dropout_rate))
            current_dim = hidden_dim
        
        # Add final classification layer
        self.fc_final = nn.Linear(current_dim, num_classes)
        
    def forward(self, x):
        # Apply convolutional blocks
        for conv, bn, pool in zip(self.conv_blocks, self.bn_layers, self.pool_layers):
            x = conv(x)
            x = bn(x)
            x = self.activation_fn(x)
            x = pool(x)
        
        # Flatten the feature maps
        x = x.view(-1, self.fc_input_dim)
        
        # Apply fully connected layers
        for fc, bn, dropout in zip(self.fc_layers, self.fc_bn_layers, self.fc_dropouts):
            x = fc(x)
            x = bn(x)
            x = self.activation_fn(x)
            x = dropout(x)
        
        # Final classification layer
        x = self.fc_final(x)
        
        return x

# Example usage:
"""
model = FlexibleCNN(
    input_channels=6,
    num_classes=5,
    window_length=100,
    conv_layers=4,
    kernel_size=9,
    hidden_layers=[500, 64],
    dropout_rate=0.5,
    activation_fn=F.relu,
    initial_filters=32
)
"""



def train_model_with_advanced_logging(
    model, 
    train_loader, 
    val_loader, 
    device,
    model_name,  
    num_epochs=50,
    learning_rate=0.001,
    weight_decay=1e-5,
    base_log_dir='runs',
    early_stopping_patience=7,
    augmenter=None,
    class_names=None 
):
    """
    Enhanced training function with comprehensive metrics and logging
    
    Additional features:
    - Macro and weighted F1 scores
    - Confusion matrix logging
    - Model architecture and hyperparameters logging
    - Per-class performance metrics
    - Learning curves
    - Gradient flow visualization
    """
    # Create unique run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_log_dir, f'{model_name}_{timestamp}')
    log_dir = os.path.join(run_dir, 'tensorboard')
    model_dir = os.path.join(run_dir, 'models')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir)

    # Save hyperparameters
    hparams = {
        'model_name': model_name,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'num_epochs': num_epochs,
        'batch_size': next(iter(train_loader))[0].shape[0],
        'early_stopping_patience': early_stopping_patience,
        'model_architecture': str(model),
        'optimizer': 'Adam',
        'scheduler': 'ReduceLROnPlateau',
        'augmentation': str(augmenter) if augmenter else 'None'
    }
    
    with open(os.path.join(run_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hparams, f, indent=4)
    
    # Log hyperparameters to tensorboard
    writer.add_text('Hyperparameters', str(hparams))

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Initialize tracking variables
    best_val_loss = float('inf')
    best_val_f1_macro = 0.0
    epochs_without_improvement = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1_macro': [], 'train_f1_weighted': [],
        'val_loss': [], 'val_acc': [], 'val_f1_macro': [], 'val_f1_weighted': []
    }

    def log_confusion_matrix(y_true, y_pred, epoch, phase, class_names):
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        fig, ax = plt.figure(), plt.axes()
        sns.heatmap(cm, annot=False, fmt='d', ax=ax)
        # Get number of classes from the confusion matrix
        n_classes = cm.shape[0]        
        # Create default class names if none provided
        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]
            
        # Ensure we have the correct number of class names
        if len(class_names) != n_classes:
            class_names = [str(i) for i in range(n_classes)]
        
        plt.xticks(np.arange(n_classes) + 0.5, class_names, rotation=45, fontsize=8)
        plt.yticks(np.arange(n_classes) + 0.5, class_names, rotation=45, fontsize=8)
        plt.title(f'{phase} Confusion Matrix')
        writer.add_figure(f'Confusion_Matrix/{phase}', fig, epoch)
        plt.close()

    def log_gradients_and_weights(epoch):
        """Log gradient and weight statistics for each layer"""
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'gradients/{name}', param.grad, epoch)
                writer.add_histogram(f'weights/{name}', param, epoch)

    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            all_preds = []
            all_true = []
            
            for X_batch, y_batch in loader:
                if phase == 'train' and augmenter is not None:
                    X_batch = augmenter.apply(X_batch)
                
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    if phase == 'train':
                        loss.backward()
                        # Log gradient norms
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                        writer.add_scalar('Gradient_Norm/train', grad_norm, epoch)
                        optimizer.step()
                
                running_loss += loss.item() * X_batch.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(y_batch.cpu().numpy())

            # Calculate metrics
            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = accuracy_score(all_true, all_preds)
            epoch_f1_macro = f1_score(all_true, all_preds, average='macro')
            epoch_f1_weighted = f1_score(all_true, all_preds, average='weighted')

            # Log metrics
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)
            writer.add_scalar(f'F1_Macro/{phase}', epoch_f1_macro, epoch)
            writer.add_scalar(f'F1_Weighted/{phase}', epoch_f1_weighted, epoch)

            # Log confusion matrix
            log_confusion_matrix(all_true, all_preds, epoch, phase, class_names)

            # Update history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)
            history[f'{phase}_f1_macro'].append(epoch_f1_macro)
            history[f'{phase}_f1_weighted'].append(epoch_f1_weighted)

            # Print metrics
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
            print(f'{phase.capitalize()} F1 Macro: {epoch_f1_macro:.4f}, F1 Weighted: {epoch_f1_weighted:.4f}')

            if phase == 'train':
                log_gradients_and_weights(epoch)
            else:  # validation phase
                # Learning rate scheduling
                scheduler.step(epoch_loss)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

                # Model saving logic
                save_model = False
                save_reason = []

                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    epochs_without_improvement = 0
                    save_model = True
                    save_reason.append('best_loss')
                else:
                    epochs_without_improvement += 1

                if epoch_f1_macro > best_val_f1_macro:
                    best_val_f1_macro = epoch_f1_macro
                    save_model = True
                    save_reason.append('best_f1')

                if save_model:
                    for reason in save_reason:
                        save_path = os.path.join(model_dir, f'{reason}_model.pt')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': epoch_loss,
                            'f1_macro': epoch_f1_macro,
                            'accuracy': epoch_acc
                        }, save_path)
                        print(f'Saved model for {reason} with val_loss: {epoch_loss:.4f}')

                # Early stopping
                if epochs_without_improvement >= early_stopping_patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    training_time = time.time() - start_time
                    writer.add_text('Training_Summary', 
                                  f'Training stopped early after {epoch+1} epochs. '
                                  f'Best val_loss: {best_val_loss:.4f}, '
                                  f'Best F1 macro: {best_val_f1_macro:.4f}. '
                                  f'Total time: {training_time:.2f} seconds')
                    writer.close()
                    return model, history

        print('-' * 60)

    # Training completed normally
    training_time = time.time() - start_time
    writer.add_text('Training_Summary', 
                   f'Training completed after {num_epochs} epochs. '
                   f'Best val_loss: {best_val_loss:.4f}, '
                   f'Best F1 macro: {best_val_f1_macro:.4f}. '
                   f'Total time: {training_time:.2f} seconds')
    writer.close()

    return model, history



def objective(trial: Trial, train_loader, val_loader, device, input_channels, num_classes, window_length, class_names = None):
    """
    Optuna objective function for hyperparameter optimization
    """
    # Define hyperparameter search space
    params = {
        'conv_layers': trial.suggest_int('conv_layers', 2, 5),
        'kernel_size': trial.suggest_int('kernel_size', 3, 15, step=2),  # Odd numbers only
        'initial_filters': trial.suggest_int('initial_filters', 16, 256, step=16),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.8),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'hidden_layers': [
            trial.suggest_int(f'hidden_layer_{i}', 32, 512)
            for i in range(trial.suggest_int('n_hidden_layers', 1, 5))
        ]
    }
    
    # Create model with trial parameters
    model = FlexibleCNN(
        input_channels=input_channels,
        num_classes=num_classes,
        window_length=window_length,
        conv_layers=params['conv_layers'],
        kernel_size=params['kernel_size'],
        hidden_layers=params['hidden_layers'],
        dropout_rate=params['dropout_rate'],
        initial_filters=params['initial_filters']
    ).to(device)
    
    # Create unique name for this trial
    trial_name = f"trial_{trial.number}_conv{params['conv_layers']}_ker{params['kernel_size']}"
    
    # try:
    # Train model
    model, history = train_model_with_advanced_logging(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        model_name=trial_name,
        learning_rate=params['learning_rate'],
        weight_decay=params['weight_decay'],
        num_epochs=30,  
        early_stopping_patience=5,
        class_names= class_names
    )
    
    # Get best validation metrics
    best_val_f1_macro = max(history['val_f1_macro'])
    best_val_loss = min(history['val_loss'])
    
    # Log metrics for this trial
    trial.set_user_attr('best_val_f1_macro', best_val_f1_macro)
    trial.set_user_attr('best_val_loss', best_val_loss)
    trial.set_user_attr('hidden_layers', params['hidden_layers'])
    
    return best_val_f1_macro 
        
    # except Exception as e:
    #     print(f"Trial {trial.number} failed with error: {str(e)}")
    #     raise optuna.exceptions.TrialPruned()

# Create a function to run the hyperparameter tuning
def run_hyperparameter_tuning(train_loader, val_loader, device, input_channels, num_classes, window_length, 
                            n_trials=50, study_name=None, class_names = None):
    """
    Run hyperparameter optimization and return the best model configuration
    """
    if study_name is None:
        study_name = f"cnn_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial, 
            train_loader, 
            val_loader, 
            device, 
            input_channels, 
            num_classes, 
            window_length,
            class_names
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    return study

# Function to analyze and visualize results
def analyze_study_results(study):
    """
    Analyze and display the results of the hyperparameter optimization
    """
    print("\nBest trial:")
    trial = study.best_trial
    
    print(f"\nBest F1 Macro Score: {trial.value:.4f}")
    print("\nBest hyperparameters:")
    
    # Print all parameters of the best trial
    for key, value in trial.params.items():
        if key == 'hidden_layers':
            print(f"  {key}:")
            for i, size in enumerate(value):
                print(f"    Layer {i+1}: {size} units")
        else:
            print(f"  {key}: {value}")
    
    # Get additional metrics from user attributes
    print(f"\nBest validation loss: {trial.user_attrs['best_val_loss']:.4f}")
    
    # Create importance plot
    optuna.visualization.plot_param_importances(study)
    
    # Create optimization history plot
    optuna.visualization.plot_optimization_history(study)
    
    return trial.params

