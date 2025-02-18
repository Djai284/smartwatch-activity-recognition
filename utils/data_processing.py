import torch
from torch.utils.data import Dataset
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

class SignalPreprocessor:
    def __init__(self, target_sampling_rate=50):
        self.target_sampling_rate = target_sampling_rate
        self.scaler = StandardScaler()
        
    def resample_signal(self, data, original_rate):
        if original_rate == self.target_sampling_rate:
            return data
            
        num_samples = int(len(data) * self.target_sampling_rate / original_rate)
        return signal.resample(data, num_samples)
    
    def normalize_signals(self, data):
        shape = data.shape
        flattened = data.reshape(-1, shape[-1])
        normalized = self.scaler.fit_transform(flattened)
        return normalized.reshape(shape)
    
    def filter_signals(self, data, lowcut=0.5, highcut=20.0):
        nyquist = self.target_sampling_rate * 0.5
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, data, axis=0)
    
    def process_sample(self, sample, original_rate):
        if original_rate != self.target_sampling_rate:
            sample = self.resample_signal(sample, original_rate)
        sample = self.filter_signals(sample)
        sample = self.normalize_signals(sample)
        return sample

class ExerciseDataset(Dataset):
    """Dataset class for exercise data"""
    def __init__(self, X, y, sampling_rates, preprocessor=None, augmenter=None):
        self.X_raw = X
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.y = torch.tensor(y_encoded, dtype=torch.long)
        
        self.sampling_rates = sampling_rates
        self.preprocessor = preprocessor
        self.augmenter = augmenter
        
        # Preprocess all samples
        self.X = []
        for i, sample in enumerate(X):
            processed = preprocessor.process_sample(
                sample, sampling_rates[i]) if preprocessor else sample
            self.X.append(processed)
        self.X = torch.tensor(np.stack(self.X), dtype=torch.float32).permute(0, 2, 1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augmenter:
            x = self.augmenter.apply(x.unsqueeze(0)).squeeze(0)
        return x, self.y[idx]
    

class MemoryEfficientDataset(Dataset):
    """Memory efficient dataset with consistent shape handling"""
    def __init__(self, data_df, indices, preprocessor=None, augmenter=None, target_length=500):
        self.data_df = data_df.iloc[indices]
        self.preprocessor = preprocessor
        self.augmenter = augmenter
        self.target_length = target_length
        self.num_channels = 6  # Fixed number of channels (3 acc + 3 gyro)
        
        # Store sampling rates mapping
        self.sampling_rates = self.data_df['dataset'].map({
            'mmfit': 100,
            'har_data': 100,
            'reco': 50
        }).values
        
        # Encode labels once
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.data_df['activity_name'].values)
    
    def process_sequence(self, sequence):
        """Process sequence to ensure consistent shape"""
        # Convert to numpy array if not already
        sequence = np.asarray(sequence)
        
        # Ensure 2D array with shape [num_channels, time_steps]
        if len(sequence.shape) == 1:
            sequence = sequence.reshape(1, -1)
        
        # Ensure correct number of channels
        if sequence.shape[0] != self.num_channels:
            if sequence.shape[1] == self.num_channels:
                sequence = sequence.T
            else:
                raise ValueError(f"Invalid number of channels: {sequence.shape}")
        
        current_length = sequence.shape[1]
        
        if current_length > self.target_length:
            # Truncate from the center
            start_idx = (current_length - self.target_length) // 2
            return sequence[:, start_idx:start_idx + self.target_length]
        elif current_length < self.target_length:
            # Pad with zeros on both ends
            padding_left = (self.target_length - current_length) // 2
            padding_right = self.target_length - current_length - padding_left
            return np.pad(sequence, 
                        ((0, 0), (padding_left, padding_right)),
                        mode='constant', 
                        constant_values=0)
        else:
            return sequence
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        # Load single sample
        sample = self.data_df.iloc[idx]['sig_array']
        label = self.labels[idx]
        
        # Preprocess if needed
        if self.preprocessor:
            sample = self.preprocessor.process_sample(
                sample, 
                self.sampling_rates[idx]
            )
        
        # Process sequence to ensure consistent shape
        sample = self.process_sequence(sample)
        
        # Convert to tensor with correct shape [num_channels, time_steps]
        x = torch.tensor(sample, dtype=torch.float32)
        
        # Apply augmentation if needed
        if self.augmenter:
            x = self.augmenter.apply(x.unsqueeze(0)).squeeze(0)
            
        # Ensure label is a tensor
        y = torch.tensor(label, dtype=torch.long)
        
        return x, y

