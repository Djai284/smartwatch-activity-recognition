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
            padding=0
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=9,
            stride=1,
            padding=0
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2)


        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=9,
            stride=1,
            padding=0
        )
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # After two conv+pool steps, the time dimension is smaller.
        # We can compute final length if we want to fully specify the linear layer size
        # (window_length -> after conv1 -> after pool1 -> after conv2 -> after pool2).
        # For simplicity, let's do a quick calculation here:

        def conv_out_size(in_size, kernel_size=5, stride=1, padding=0):
            return (in_size - kernel_size + 2*padding)//stride + 1

        # conv1
        out1 = conv_out_size(window_length, 9, 1, 0)
        out1_pool = out1 // 2  # after maxpool1d(kernel_size=2)

        # conv2
        out2 = conv_out_size(out1_pool, 9, 1, 0)
        out2_pool = out2 // 2

        # conv 3
        out3 = conv_out_size(out2_pool, 9, 1, 0)
        out3_pool = out3 // 2

        # The final feature dimension = out_channels(=128) * out2_pool
        self.fc_input_dim = 128 * out3_pool

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, 500)
        self.dropout1 = nn.Dropout(p=0.33)
        self.fc2 = nn.Linear(500, 64)
        self.drouput2 = nn.Dropout(p=0.33)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        x shape: [batch_size, 6, window_length]
        """
        # 1) First convolution + ReLU + max pool
        x = F.relu(self.conv1(x))     # -> [batch_size, 64, out1]
        x = self.pool1(x)            # -> [batch_size, 64, out1_pool]

        # 2) Second convolution + ReLU + max pool
        x = F.relu(self.conv2(x))     # -> [batch_size, 128, out2]
        x = self.pool2(x)            # -> [batch_size, 128, out2_pool]

        # 2) Second convolution + ReLU + max pool
        x = F.relu(self.conv3(x))     # -> [batch_size, 128, out2]
        x = self.pool3(x)            # -> [batch_size, 128, out2_pool]

        # 3) Flatten the feature maps
        x = x.view(-1, self.fc_input_dim)  # flatten to [batch_size, fc_input_dim]


        # 4) Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.drouput2(x)
        x = self.fc3(x)  # final logits


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
    output = model(dummy_input)
    print("Output shape:", output.shape)  # should be [8, num_classes]
