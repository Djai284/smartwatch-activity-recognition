import torch
import torch.nn as nn
import torch.nn.functional as F
# training.py
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score

def train_model_with_tensorboard(model, train_loader, val_loader, device, 
                               num_epochs=50, 
                               learning_rate=0.001, 
                               weight_decay=1e-5,
                               log_dir='runs/exercise_cnn',
                               save_dir='models',
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


# class IMUCNN(nn.Module):
#     def __init__(self, num_classes=5, window_length=100):
#         super().__init__()
#         # Convolution layers
#         self.conv1 = nn.Conv1d(
#             in_channels=6,   # we have 6 input channels: (accX, accY, accZ, gyrX, gyrY, gyrZ)
#             out_channels=64, # number of filters
#             kernel_size=9,   # how many time-steps the kernel covers
#             stride=1,
#             padding=4        # Changed to 'same' padding (kernel_size//2)
#         )
#         self.bn1 = nn.BatchNorm1d(64)  # Add batch normalization
#         self.pool1 = nn.MaxPool1d(kernel_size=2)

#         self.conv2 = nn.Conv1d(
#             in_channels=64,
#             out_channels=128,
#             kernel_size=9,
#             stride=1,
#             padding=4        # Same padding
#         )
#         self.bn2 = nn.BatchNorm1d(128)  # Add batch normalization
#         self.pool2 = nn.MaxPool1d(kernel_size=2)

#         self.conv3 = nn.Conv1d(
#             in_channels=128,
#             out_channels=128,
#             kernel_size=9,
#             stride=1,
#             padding=4        # Same padding
#         )
#         self.bn3 = nn.BatchNorm1d(128)  # Add batch normalization
#         self.pool3 = nn.MaxPool1d(kernel_size=2)

#         # Calculate output size after convolutions and pooling
#         # With 'same' padding, only pooling reduces the size
#         out_size = window_length
#         out_size = out_size // 2  # after pool1
#         out_size = out_size // 2  # after pool2
#         out_size = out_size // 2  # after pool3
        
#         # The final feature dimension
#         self.fc_input_dim = 128 * out_size

#         # Fully connected layers
#         self.fc1 = nn.Linear(self.fc_input_dim, 500)
#         self.bn_fc1 = nn.BatchNorm1d(500)  # Add batch normalization
#         self.dropout1 = nn.Dropout(p=0.5)  # Increased dropout
        
#         self.fc2 = nn.Linear(500, 64)
#         self.bn_fc2 = nn.BatchNorm1d(64)   # Add batch normalization
#         self.dropout2 = nn.Dropout(p=0.5)  # Fix typo and increase dropout
        
#         self.fc3 = nn.Linear(64, num_classes)

#     def forward(self, x):
#         """
#         x shape: [batch_size, 6, window_length]
#         """
#         # 1) First convolution + BatchNorm + ReLU + max pool
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.pool1(x)

#         # 2) Second convolution + BatchNorm + ReLU + max pool
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = self.pool2(x)

#         # 3) Third convolution + BatchNorm + ReLU + max pool
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = F.relu(x)
#         x = self.pool3(x)

#         # 4) Flatten the feature maps
#         x = x.view(-1, self.fc_input_dim)

#         # 5) Dense layers with BatchNorm
#         x = self.fc1(x)
#         x = self.bn_fc1(x)
#         x = F.relu(x)
#         x = self.dropout1(x)
        
#         x = self.fc2(x)
#         x = self.bn_fc2(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
        
#         x = self.fc3(x)  # final logits

#         return x

class IMUCNN(nn.Module):
    def __init__(self, num_classes=5, window_length=100):
        super().__init__()
        # Convolution layers
        self.conv1 = nn.Conv1d(
            in_channels=6,   # we have 6 input channels: (accX, accY, accZ, gyrX, gyrY, gyrZ)
            out_channels=64, # number of filters
            kernel_size=9,   # how many time-steps the kernel covers
            stride=1,
            padding=4        # Changed to 'same' padding (kernel_size//2)
        )
        self.bn1 = nn.BatchNorm1d(64)  # Add batch normalization
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=9,
            stride=1,
            padding=4        # Same padding
        )
        self.bn2 = nn.BatchNorm1d(128)  # Add batch normalization
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=9,
            stride=1,
            padding=4        # Same padding
        )
        self.bn3 = nn.BatchNorm1d(128)  # Add batch normalization
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Calculate output size after convolutions and pooling
        # With 'same' padding, only pooling reduces the size
        out_size = window_length
        out_size = out_size // 2  # after pool1
        out_size = out_size // 2  # after pool2
        out_size = out_size // 2  # after pool3
        
        # The final feature dimension
        self.fc_input_dim = 128 * out_size

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, 500)
        self.bn_fc1 = nn.BatchNorm1d(500)  # Add batch normalization
        self.dropout1 = nn.Dropout(p=0.5)  # Increased dropout
        
        self.fc2 = nn.Linear(500, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)   # Add batch normalization
        self.dropout2 = nn.Dropout(p=0.5)  # Fix typo and increase dropout
        
        self.fc3 = nn.Linear(64, num_classes)

        # Add ReLU modules for quantization
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.relu3 = nn.ReLU(inplace=False)

    def fuse_model(self):
        """Fuse operations for better quantization"""
        torch.quantization.fuse_modules(
            self,
            [
                ['conv1', 'bn1', 'relu1'],
                ['conv2', 'bn2', 'relu2'],
                ['conv3', 'bn3', 'relu3']
            ],
            inplace=True
        )

    def prepare_for_quantization(self):
        """Prepare model for quantization"""
        # Replace ReLU with quantization-friendly version
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.relu3 = nn.ReLU(inplace=False)
        # Fuse layers
        self.fuse_model()

    def forward(self, x):
        """
        x shape: [batch_size, 6, window_length]
        """
        # 1) First convolution + BatchNorm + ReLU + max pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)  # Use class ReLU instead of F.relu
        x = self.pool1(x)

        # 2) Second convolution + BatchNorm + ReLU + max pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)  # Use class ReLU instead of F.relu
        x = self.pool2(x)

        # 3) Third convolution + BatchNorm + ReLU + max pool
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)  # Use class ReLU instead of F.relu
        x = self.pool3(x)

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
        
        x = self.fc3(x)  # final logits

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


# Example usage:
if __name__ == "__main__":
    # Suppose a batch of 8 examples, each with window_length=100 time steps, 6 channels
    batch_size = 8
    window_length = 100
    num_classes = 5

    model = IMUCNN(num_classes=num_classes, window_length=window_length)
    print(model)

    # Create a dummy input
    dummy_input = torch.randn(batch_size, 6, window_length)  # [8, 6, 100]
    
    # Apply augmentation (during training only)
    augmenter = TimeSeriesAugmentation()
    augmented_input = augmenter.apply(dummy_input)
    
    output = model(augmented_input)
    print("Output shape:", output.shape)  # should be [8, num_classes]