import torch
import torch.nn as nn
import torch.nn.functional as F

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