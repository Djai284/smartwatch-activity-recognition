import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

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

# Define or import your FlexibleCNN class
# If FlexibleCNN is defined in a separate file (e.g., model.py), import it:
# from model import FlexibleCNN

# Instantiate your model with the parameters used during training.
# Adjust these values as needed.
model = FlexibleCNN(
    input_channels=6,
    num_classes=34,
    window_length=288,         # Make sure this matches your training configuration.
    conv_layers=4,
    kernel_size=7,
    hidden_layers=[428, 351],   # Example hidden layers; update if different.
    dropout_rate=0.5,
    activation_fn=F.relu,
    initial_filters=64
)

# Load the state dictionary saved in best_f1_model.pt
state_dict = torch.load("./runs/final_model/models/best_f1_model.pt", map_location=torch.device("cpu"))
model.load_state_dict(state_dict["model_state_dict"])
model.eval()

# Create an example input matching your model's expected input shape.
# Here, batch size is 1, 6 channels, and a window length of 100.
example_input = torch.randn(1, 6, 288)

# Trace the model using TorchScript.
traced_model = torch.jit.trace(model, example_input)

# Convert the traced model to Core ML.
mlmodel = ct.convert(traced_model, inputs=[ct.TensorType(shape=example_input.shape)])

# Save the Core ML model.
mlmodel.save("BestF1Model.mlpackage")

print("Conversion complete! Saved as BestF1Model.mlmodel")
