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