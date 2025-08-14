import numpy as np
import argparse
import time
import csv
from tqdm import tqdm
import pandas as pd
import os
import json
import re
import numpy as np
import pandas as pd
import scipy.io
import warnings
from typing import Dict, Tuple, Any, Optional, List
import logging
from scipy import signal

from utils.constants import rename_dict, name_fix_dict



def load_mmfit(filepath = 'mm-fit', r = 21):

       raw_data = pd.DataFrame(columns = ['index', 'frame', 'timestamp', 'acc_X', 'acc_Y', 'acc_Z', 'gyr_X',
              'gyr_Y', 'gyr_Z', 'activity_name', 'subject_id'])

       for i in tqdm(range(r)):

              acc_data = load_modality(f'{filepath}/w{i:02}/w{i:02}_sw_l_acc.npy')
              gyr_data = load_modality(f'{filepath}/w{i:02}/w{i:02}_sw_l_gyr.npy')
              label_data = pd.read_csv(f'{filepath}/w{i:02}/w{i:02}_labels.csv', names = ['start_frame', 'end_frame', 'repetitions', 'activity_name'])

              df = create_df(acc_data, gyr_data, label_data, i)
              raw_data = pd.concat([raw_data, df], axis =0)

       return raw_data

def find_key_by_value(dictionary, filename):
    # Extract numeric part from the filename
    match = re.search(r'(\d+)', filename)
    if not match:
        return None  
    
    num = int(match.group(1))  # Convert extracted number to integer

    # Search for the key containing this number in its list of values
    for key, values in dictionary.items():
        if num in values:
            return key

    return None 

def load_crossfit(datapath = None, info_path=None):

    if not datapath:
        datapath = f"{os.getcwd()}/np_exercise_data"

    if not info_path:
        info_path = 'participant_ex_code_map.txt'

    with open(info_path, 'rb') as f:
        part_info = json.load(f)

    df = pd.DataFrame(columns = ['acc_X', 'acc_Y', 'acc_Z', 'gyr_X', 'gyr_Y', 'gyr_Z', 'subject_id', 'activity_name'])

    for root, dirs, files in os.walk(datapath):

        # loading different exercise data
        for dir in tqdm(dirs):

            # loop through the files
            for root, dirs1, files1 in os.walk(datapath+'/'+dir):

                # loop through all the files
                for file in files1:


                    data = np.load(f'{datapath}/{dir}/{file}')
                    temp_df = pd.DataFrame(data[:6, :].T, columns = ['acc_X', 'acc_Y', 'acc_Z', 'gyr_X', 'gyr_Y', 'gyr_Z'])
                    temp_df['activity_name'] = dir.lower().replace(' ', '')
                    temp_df['subject_id'] = find_key_by_value(part_info, file)

                    df = pd.concat([df, temp_df], axis = 0)

    return df


import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # For progress tracking

def create_examples(df, dim=500, between=None, show_progress=True):
    """
    This creates the data in proper form for a CNN model
    Args:
        df: The data in format one reading per row 
        dim: dimension of the cnn (number of reading in each input)
        between: time between starts of different sequence (if none they will be completely separate)
        show_progress: Whether to show a progress bar
    """
    # Convert to numpy for faster operations
    activity_array = df['activity_name'].values
    subject_array = df['subject_id'].values
    dataset_array = df['dataset'].values
    sensor_columns = ['gyr_X', 'gyr_Y', 'gyr_Z', 'acc_X', 'acc_Y', 'acc_Z']
    sensor_data = df[sensor_columns].values
    repetitions_array = df['repetitions'].values

    
    result_ids = []
    result_sig_arrays = []
    result_activity_names = []
    result_subject_ids = []
    result_datasets = []
    result_repetitions = []
    result_switch_blocks = []
    
    print("Creating dataset for model...")
    
    i = 0
    total_len = len(df)
    
    # Set up progress bar if requested
    pbar = tqdm(total=total_len) if show_progress else None
    
    while i < total_len:
        user = subject_array[i]
        dataset = dataset_array[i]
        label = activity_array[i]
        repetitions = repetitions_array[i]
        switch_block = None
        
        # Find the end of this segment (where activity or user changes)
        segment_end = i
        # while segment_end < total_len and activity_array[segment_end] == label and subject_array[segment_end] == user:
        while segment_end < total_len and subject_array[segment_end] == user and (dataset_array[segment_end] != 'har_data' or activity_array[segment_end] == label):
            segment_end += 1
        
        # Process this segment
        j = i
        while j + dim <= segment_end:

            if activity_array[j+dim] != activity_array[j]:

                # find where the next activity starts
                k = j
                while activity_array[k] == activity_array[j]:
                    k += 1

                switch_block = k - j

                if (k-j)/dim <=.5:
                    label = activity_array[k]
                    repetitions = repetitions_array[k] if repetitions_array[k] else 0


            # Extract the chunk of sensor data
            arr = sensor_data[j:j+dim].copy()
            
            # Add to results
            result_ids.append(len(result_ids))
            result_sig_arrays.append(arr)
            result_activity_names.append(label)
            result_subject_ids.append(user)
            result_datasets.append(dataset)
            result_switch_blocks.append(switch_block)
            result_repetitions.append(repetitions)
            
            # Move to next position
            if between is None:
                j += dim  # Non-overlapping windows
            else:
                j += between  # Overlapping windows with specified step

            if switch_block:
                switch_block -= between
                if switch_block < 0:
                    switch_block = None

        
        # Update progress bar
        if pbar is not None:
            pbar.update(segment_end - i)
        
        # Move to the next segment
        i = segment_end
    
    if pbar is not None:
        pbar.close()
    
    # Create dataframe from results
    ex_df = pd.DataFrame({
        'id': result_ids,
        'sig_array': result_sig_arrays,
        'activity_name': result_activity_names,
        'subject_id': result_subject_ids,
        'dataset': result_datasets,
        'repetitions': result_repetitions,
        'switch_block': result_switch_blocks,
    })
    
    print(f"Created {len(ex_df)} examples")
    return ex_df

def process_chunk(args):
    """Helper function for parallel processing with progress tracking"""
    chunk_df, dim, between, chunk_id, total_chunks = args
    result = create_examples(chunk_df, dim, between, show_progress=False)
    print(f"Processed chunk {chunk_id+1}/{total_chunks} with {len(result)} examples")
    return result

def create_examples_parallel(df, dim=500, between=None, n_workers=4):
    """Parallel version of create_examples using multiple processes"""
    # Split dataframe into chunks by subject and activity
    chunks = []
    current_chunk = []
    
    for i in range(len(df)):
        if i == 0:
            current_chunk = [i]
        elif (df.iloc[i]['subject_id'] != df.iloc[i-1]['subject_id']):
            chunks.append((current_chunk[0], i))
            current_chunk = [i]
    
    if current_chunk:
        chunks.append((current_chunk[0], len(df)))
    
    # Process chunks in parallel
    df_chunks = [df.iloc[start:end].copy().reset_index(drop=True) for start, end in chunks]
    print(f"Split data into {len(df_chunks)} chunks for parallel processing")
    
    # Prepare arguments with chunk IDs for progress tracking
    chunk_args = [(chunk, dim, between, i, len(df_chunks)) 
                  for i, chunk in enumerate(df_chunks)]
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_chunk, chunk_args))
    
    # Combine results
    combined_df = pd.concat(results, ignore_index=True)
    combined_df['id'] = range(len(combined_df))  # Fix IDs
    
    print(f"Final dataset contains {len(combined_df)} examples")
    return combined_df


def create_df(acc_data, gyr_data, label_df, person_id = 0):

    # create dataframes for acceleromters and gyrosccopes
    acc_df = pd.DataFrame(acc_data, columns=['frame', 'timestamp', 'acc_X', 'acc_Y', 'acc_Z']).reset_index()
    gyr_df = pd.DataFrame(gyr_data, columns=['frame', 'timestamp', 'gyr_X', 'gyr_Y', 'gyr_Z']).reset_index()

    # combine the two dataframes
    df = pd.merge(acc_df, gyr_df, how = 'inner', on = ['index', 'frame', 'timestamp'])
    df['activity_name'] = " "
    df['repetitions'] = 0
    df['subject_id'] = person_id

    # start at the begnining of the labels
    label_tracker = 0
    repetitions = 0
    label = label_df.loc[label_tracker, 'activity_name']
    repetitions = label_df.loc[label_tracker, 'repetitions']
    
    # loop througbn the dataframe
    for i, row in df.iterrows():
        
        # move to next label if frame is past current exercise
        if row['frame'] > label_df.loc[label_tracker, 'end_frame']:
            label_tracker += 1

            # move to next exercise
            if label_tracker < len(label_df):
                label = label_df.loc[label_tracker, 'activity_name']
                repetitions = label_df.loc[label_tracker, 'repetitions']

            # break from loop if no more exercises
            else:
                df.loc[i:, 'activity_name'] = 'non-e'
                break
        
        # assign proper label
        if row['frame'] > label_df.loc[label_tracker, 'start_frame']:
            df.at[i, 'activity_name'] = label
            df.at[i, 'repetitions'] = repetitions
        else:
            df.at[i, 'activity_name'] = "non-e"

    return df



def load_modality(filepath):
    """
    Loads modality from filepath and returns numpy array, or None if no file is found.
    :param filepath: File path to MM-Fit modality.
    :return: MM-Fit modality (numpy array).
    """
    try:
        mod = np.load(filepath)
    except FileNotFoundError as e:
        mod = None
        print('{}. Returning None'.format(e))
    return mod


def load_labels(filepath):
    """
    Loads and reads CSV MM-Fit CSV label file.
    :param filepath: File path to a MM-Fit CSV label file.
    :return: List of lists containing label data, (Start Frame, End Frame, Repetition Count, Activity) for each
    exercise set.
    """
    labels = []
    with open(filepath, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            labels.append([int(line[0]), int(line[1]), int(line[2]), line[3]])
    return labels


def get_subset(data, start=0, end=None):
    """
    Returns a subset of modality data.
    :param data: Modality (numpy array).
    :param start: Start frame of subset.
    :param end: End frame of subset.
    :return: Subset of data (numpy array).
    """
    if data is None:
        return None

    # Pose data
    if len(data.shape) == 3:
        if end is None:
            end = data[0, -1, 0]
        return data[:, np.where(((data[0, :, 0]) >= start) & ((data[0, :, 0]) <= end))[0], :]

    # Accelerometer, gyroscope, magnetometer and heart-rate data
    else:
        if end is None:
            end = data[-1, 0]
        return data[np.where((data[:, 0] >= start) & (data[:, 0] <= end)), :][0]


def parse_args():
    """
    Parse command-line arguments to train and evaluate a multimodal network for activity recognition on MM-Fit.
    :return: Populated namespace.
    """
    parser = argparse.ArgumentParser(description='MM-Fit Demo')
    parser.add_argument('--data', type=str, default='mm-fit/',
                        help='location of the dataset')
    parser.add_argument('--unseen_test_set', default=False, action='store_true',
                        help='if set to true the unseen test set is used for evaluation')
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='how often to eval model (in epochs)')
    parser.add_argument('--early_stop', type=int, default=20,
                        help='stop after this number of epoch if the validation loss did not improve')
    parser.add_argument('--checkpoint', type=int, default=10,
                        help='how often to checkpoint model parameters (epochs)')
    parser.add_argument('--multimodal_ae_wp', type=str, default='',
                        help='file path for the weights of the multimodal autoencoder part of the model')
    parser.add_argument('--model_wp', type=str, default='',
                        help='file path for weights of the full model')
    parser.add_argument('--window_length', type=int, default=5,
                        help='length of data window in seconds')
    parser.add_argument('--window_stride', type=float, default=0.2,
                        help='length of window stride in seconds')
    parser.add_argument('--target_sensor_sampling_rate', type=float, default=50,
                        help='Sampling rate of sensor input signal (Hz)')
    parser.add_argument('--skeleton_sampling_rate', type=float, default=30,
                        help='sampling rate of input skeleton data (Hz)')
    parser.add_argument('--layers', type=int, default=3,
                        help='number of FC layers')
    parser.add_argument('--hidden_units', type=int, default=200,
                        help='number of hidden units')
    parser.add_argument('--ae_layers', type=int, default=3,
                        help='number of autoencoder FC layers')
    parser.add_argument('--ae_hidden_units', type=int, default=200,
                        help='number of autoencoder hidden units')
    parser.add_argument('--embedding_units', type=int, default=100,
                        help='number of hidden units')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout percentage')
    parser.add_argument('--ae_dropout', type=float, default=0.0,
                        help='multimodal autoencoder dropout percentage')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='number of output classes')
    parser.add_argument('--name', type=str, default='mmfit_demo_' + str(int(time.time())),
                        help='name of experiment')
    parser.add_argument('--output', type=str, default='output/',
                        help='path to output folder')
    return parser.parse_args()


"""
Utility functions for ML model inference and exercise analysis.

This module contains all the supporting functions needed for exercise
recognition, segment analysis, and repetition counting.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from pathlib import Path
import os
import pickle
import json
from typing import Dict, List, Tuple, Optional
from scipy.signal import savgol_filter, find_peaks
from sklearn.decomposition import PCA


# Data loading and preprocessing functions
def load_pretrained_model(model_location):
    """Load the pretrained model from config and state dict."""
    model_path = Path(model_location)
    
    # Look for hyperparameters.json in the same directory
    config_path = str(model_path) + "/hyperparameters.json"
    
    print(f"Loading model config from {config_path}")
    
    # Import model loading function (assuming it's available)
    try:
        from model import load_model_from_config
    except ImportError:
        raise ImportError("Could not import load_model_from_config from model module")
    
    model = load_model_from_config(str(config_path))
    
    # Load state dict
    print(f"Loading model weights from {model_location}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    state_dict = torch.load(model_location + "/models/best_model.pt", map_location=device)
    
    # Handle the case where model is returned as a list/tuple
    if isinstance(model, (list, tuple)):
        model = model[0]
    
    model.load_state_dict(state_dict)
    return model, device
def process_custom_data(file_paths, clip=(100, 100), save=False, save_name=None, subject_id=None, window = 300, spacer = 50):
    """
    Process custom CSV data files in the same format as jai_data, jacob_data, and jack_data.
    
    Parameters:
    -----------
    file_paths : list of str
        List of file paths to CSV files to be processed
    clip : tuple of 2 ints, default (100, 100)
        Number of data points to clip from the front and back of each exercise
    save : bool, default False
        Whether to save the processed data as a pickle file
    save_name : str, optional
        File path to save the data. If None and save=True, generates a name based on input files
    subject_id : str, optional
        Name of the subject. If None, uses the first word of the first filename
    
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with standardized columns and cleaned activity names
    """
    
    
    # Define the trimming function from the notebook
    def trim_label_group(group):
        return group.iloc[clip[0]:-clip[1]] if len(group) > (clip[0] + clip[1]) else group.iloc[0:0]
    
    # Load and process each CSV file
    data_list = []
    
    for file_path in file_paths:
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Set subject_id if not provided
        if subject_id is None:
            # Extract first word from filename
            filename = Path(file_path).stem
            current_subject_id = filename.split('_')[0]  # Gets first part before underscore
        else:
            current_subject_id = subject_id
            
        df['subject_id'] = current_subject_id
        data_list.append(df)
    
    
    # Concatenate all data
    custom_data = pd.concat(data_list, axis=0, ignore_index=True)

    # Apply trimming to each label group
    if clip[0] > 0 or clip[1] > 0:
        custom_data = custom_data.groupby('label', group_keys=False).apply(trim_label_group)
    custom_data = custom_data.reset_index(drop=True)

    # Rename columns
    custom_data.rename(columns=rename_dict, inplace=True)
    
    # Add dataset column
    custom_data['dataset'] = 'custom'
    
    # Standardize sensor data
    to_scale = ['acc_X', 'acc_Y', 'acc_Z', 'gyr_X', 'gyr_Y', 'gyr_Z']
    
    # Only scale columns that exist in the dataframe
    columns_to_scale = [col for col in to_scale if col in custom_data.columns]
    if columns_to_scale:
        scaler = StandardScaler()
        custom_data.loc[:, columns_to_scale] = scaler.fit_transform(custom_data.loc[:, columns_to_scale])
    
    # Apply name mapping and cleaning to predicted exercise
    if 'activity_name' in custom_data.columns:
        custom_data['activity_name'] = custom_data['activity_name'].apply(
            lambda x: name_fix_dict.get(x, x) if x in name_fix_dict else x
        )
        custom_data['activity_name'] = custom_data['activity_name'].apply(
            lambda x: str(x).lower().replace(" ", '')
        )
        custom_data['activity_name'] = custom_data['activity_name'].apply(
            lambda x: name_fix_dict.get(x, x) if x in name_fix_dict else x
        )

    # Apply same cleaning to actual exercise if it exists
    if 'actual_activity_name' in custom_data.columns:
        custom_data['actual_activity_name'] = custom_data['actual_activity_name'].apply(
            lambda x: name_fix_dict.get(x, x) if pd.notna(x) else x
        )
        custom_data['actual_activity_name'] = custom_data['actual_activity_name'].apply(
            lambda x: str(x).lower().replace(" ", '') if pd.notna(x) else x
        )
        custom_data['actual_activity_name'] = custom_data['actual_activity_name'].apply(
            lambda x: name_fix_dict.get(x, x) if pd.notna(x) else x
        )

    # Handle repetitions - use actual_reps if available, otherwise default to 0
    if 'repetitions' not in custom_data.columns:
        custom_data['repetitions'] = 0
        
    # Ensure required columns exist with defaults
    required_columns = {
        'user_id': 'unknown',
        'workout_id': 'unknown', 
        'actual_activity_name': None,
        'heart_rate': None
    }

    for col, default_val in required_columns.items():
        if col not in custom_data.columns:
            custom_data[col] = default_val

    custom_data = create_examples(custom_data, dim=window, between = spacer)
    
    # Save if requested
    if save:
        if save_name is None:
            # Generate save name based on input files
            base_names = [Path(fp).stem for fp in file_paths]
            combined_name = "_".join(base_names[:3])  # Limit to first 3 names to avoid too long names
            if len(base_names) > 3:
                combined_name += f"_and_{len(base_names)-3}_more"
            save_name = f"data/processed_custom_data_{combined_name}.pkl"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        
        # Save as pickle
        with open(save_name, 'wb') as f:
            pickle.dump(custom_data, f)
        print(f"Data saved to: {save_name}")
    
    return custom_data

def load_and_preprocess_data(data_location, label_encoder=None):
    """Load and preprocess the data from pickle file."""
    
    if data_location.endswith('.pkl'):
        print(f"Loading data from {data_location}")
        df = pd.read_pickle(data_location)
    elif data_location.endswith('.csv'):
        print(f"Loading data from {data_location}")
        df = process_custom_data([data_location])
    
    print(f"Loaded {len(df)} samples from data")
    
    # Filter out unwanted activities
    unwanted_activities = ['wallball', 'staticstretch', 'walk', 'repetitivestretching', 
                          'dynamicstretch(atyourownpace)', 'jumpingjacks']
    df = df[~df['activity_name'].isin(unwanted_activities)]
    
    print(f"Found {len(df['activity_name'].unique())} unique activities after filtering")
    print(f"Activity distribution:\n{df['activity_name'].value_counts()}")
    
    # Extract features and labels
    X = np.array(df['sig_array'])
    y = np.array(df['activity_name'])
    
    # Encode labels
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    else:
        y_encoded = label_encoder.transform(y)
    
    # Convert to proper format
    X = np.stack(X).astype(np.float32)
    X = torch.tensor(X, dtype=torch.float32)
    y_encoded = torch.tensor(y_encoded, dtype=torch.long)
    
    return X, y_encoded, label_encoder


def create_dataloader(X, y, batch_size=32, shuffle=True):
    """Create DataLoader from features and labels."""
    dataset = IMUDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Dataset classes
class IMUDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # Convert to tensor and swap axes
        self.y = torch.tensor(y, dtype=torch.long)  # Ensure labels are integers for classification

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class IMUDatasetWithMeta(Dataset):
    def __init__(self, X, y, subject_ids, original_indices):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # Convert to tensor and swap axes
        self.y = torch.tensor(y, dtype=torch.long)  # Ensure labels are integers for classification
        self.subject_ids = subject_ids
        self.original_indices = original_indices

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.subject_ids[idx], self.original_indices[idx]


def custom_collate_fn(batch):
    """Custom collate function to handle mixed data types"""
    X_batch = torch.stack([item[0] for item in batch])
    y_batch = torch.stack([item[1] for item in batch])
    subject_batch = [item[2] for item in batch]  # Keep as list
    index_batch = [item[3] for item in batch]    # Keep as list
    return X_batch, y_batch, subject_batch, index_batch


# Model prediction functions
def get_model_predictions_with_meta(model, test_loader_with_meta, device):
    """Get model predictions along with metadata"""
    model.eval()
    predictions = []
    true_labels = []
    subject_ids = []
    original_indices = []
    prediction_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch, subject_batch, index_batch in test_loader_with_meta:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(y_batch.numpy())
            subject_ids.extend(subject_batch)  # Already a list
            original_indices.extend(index_batch)  # Already a list
            prediction_probs.extend(probs.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels), np.array(subject_ids), np.array(original_indices), np.array(prediction_probs)


# Segment analysis functions
def find_segments(data_df, predictions, subject_ids, original_indices, label_encoder=None, 
                 confidence_threshold=0.7, use_predictions=False, prediction_probs=None, 
                 min_segment_length=5, min_rest_length=5, debug=False):
    """
    Find continuous exercise segments in the data.
    """

    
    
    rest_classes = ['rest', 'deviceontable']
    if label_encoder is not None:
        rest_indices = [list(label_encoder.classes_).index(cls) for cls in rest_classes if cls in label_encoder.classes_]
    else:
        rest_indices = rest_classes  # When predictions are strings

    if debug:
        print(f"Rest classes: {rest_classes}")
        print(f"Rest indices: {rest_indices}")
        print(f"Min rest length: {min_rest_length}")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"Use predictions: {use_predictions}")
    
    segments = []
    current_segment = None
    consecutive_rest_count = 0
    rest_buffer = []  # Store rest periods during segment
    
    # Sort by original index to maintain order
    sort_idx = np.argsort(original_indices)
    sorted_predictions = predictions[sort_idx]
    sorted_subjects = subject_ids[sort_idx]
    sorted_indices = original_indices[sort_idx]
    if prediction_probs is not None:
        sorted_probs = prediction_probs[sort_idx]
    
    for i in range(len(sorted_predictions)):
        current_pred = sorted_predictions[i]
        current_subject = sorted_subjects[i]
        current_idx = sorted_indices[i]
        current_exercise = label_encoder.classes_[current_pred] if label_encoder is not None else current_pred
        current_prob = sorted_probs[i] if prediction_probs is not None else None
        
        # Check if current prediction is rest
        if current_pred in rest_indices:
            consecutive_rest_count += 1
            
            # For predicted segments, store rest periods to potentially include them
            if use_predictions and current_segment is not None:
                rest_buffer.append({
                    'idx': current_idx,
                    'pred': current_pred,
                    'prob': current_prob,
                    'position': i
                })
            
            # Check if we should break the segment
            if consecutive_rest_count >= min_rest_length:
                if current_segment is not None:
                    segments.append(current_segment)
                    current_segment = None
                rest_buffer = []  # Clear rest buffer since we're breaking
            
            continue
        else:
            # We encountered a non-rest exercise
            
            # If we had a short rest period and are doing predictions, include the rest in segment
            if use_predictions and consecutive_rest_count > 0 and consecutive_rest_count < min_rest_length and current_segment is not None:
                # Add rest periods to current segment
                for rest_item in rest_buffer:
                    current_segment['indices'].append(rest_item['idx'])
                    current_segment['predictions'].append(rest_item['pred'])
                    if rest_item['prob'] is not None:
                        current_segment['probs'].append(rest_item['prob'])
            
            # Reset rest counter and buffer
            consecutive_rest_count = 0
            rest_buffer = []
        
        # Check if we should continue current segment or start new one
        should_continue = False
        
        if current_segment is not None:
            # Check if same subject
            if current_subject == current_segment['subject_id']:
                # For true segments, require exact exercise match
                if not use_predictions:
                    prev_label = current_segment['predictions'][-1] if current_segment['predictions'] else current_pred
                    if current_pred == prev_label:
                        should_continue = True
                else:
                    # For predicted segments, use probability-based approach if available
                    if prediction_probs is not None:
                        # Get non-rest predictions and probabilities from current segment
                        segment_exercise_probs = []
                        segment_exercise_preds = []
                        
                        for j, seg_pred in enumerate(current_segment['predictions']):
                            if seg_pred not in rest_indices:
                                segment_exercise_probs.append(current_segment['probs'][j])
                                segment_exercise_preds.append(seg_pred)
                        
                        if len(segment_exercise_probs) == 0:
                            # No exercise predictions yet, start with this one
                            should_continue = True
                        elif len(segment_exercise_probs) < 3:
                            # Not enough data to check confidence, be permissive
                            should_continue = True
                        else:
                            # Calculate cumulative probability distribution for segment
                            segment_prob_sum = np.sum(segment_exercise_probs, axis=0)
                            segment_prob_normalized = segment_prob_sum / np.sum(segment_prob_sum)
                            
                            # Find the exercise with highest total probability in segment
                            dominant_exercise_idx = np.argmax(segment_prob_normalized)
                            dominant_exercise_prob = segment_prob_normalized[dominant_exercise_idx]
                            
                            # Check current prediction's compatibility
                            current_pred_prob_for_dominant = current_prob[dominant_exercise_idx]
                            current_pred_top_idx = np.argmax(current_prob)
                            
                            # Allow continuation if conditions are met
                            segment_confident = dominant_exercise_prob >= confidence_threshold
                            current_supports_segment = current_pred_prob_for_dominant >= 0.1
                            current_matches_dominant = current_pred_top_idx == dominant_exercise_idx
                            
                            if segment_confident and current_supports_segment:
                                should_continue = True
                            elif not segment_confident:
                                should_continue = True  # Still building consensus
                            elif current_matches_dominant:
                                should_continue = True
                    else:
                        # Fallback to vote-based approach if no probabilities
                        segment_exercise_preds = [p for p in current_segment['predictions'] if p not in rest_indices]
                        
                        if len(segment_exercise_preds) == 0:
                            should_continue = True
                        elif len(segment_exercise_preds) < 3:
                            should_continue = True
                        else:
                            segment_counter = Counter(segment_exercise_preds)
                            most_common_pred, most_common_count = segment_counter.most_common(1)[0]
                            segment_confidence = most_common_count / len(segment_exercise_preds)
                            
                            current_pred_count = segment_counter.get(current_pred, 0)
                            current_pred_ratio = current_pred_count / len(segment_exercise_preds)
                            
                            if (current_pred == most_common_pred or 
                                segment_confidence < confidence_threshold or
                                current_pred_ratio >= (1 - confidence_threshold) / 2):
                                should_continue = True
        
        if should_continue:
            # Continue current segment
            current_segment['indices'].append(current_idx)
            current_segment['predictions'].append(current_pred)
            current_segment['end_idx'] = current_idx
            if current_prob is not None:
                current_segment['probs'].append(current_prob)
        else:
            # End current segment and start new one
            if current_segment is not None:
                segments.append(current_segment)
            
            current_segment = {
                'start_idx': current_idx,
                'end_idx': current_idx,
                'indices': [current_idx],
                'predictions': [current_pred],
                'subject_id': current_subject,
                'probs': [current_prob] if current_prob is not None else None
            }
    
    # Don't forget the last segment
    if current_segment is not None:
        segments.append(current_segment)
    
    # Filter segments by minimum length
    filtered_segments = [seg for seg in segments if len(seg['indices']) >= min_segment_length]
    
    return filtered_segments


def reconstruct_segment_signal(segment_indices, data_df, window_size=300, step_size=50):
    """
    Reconstruct the full signal array for a segment based on overlapping windows.
    """
    if len(segment_indices) == 1:
        return data_df.loc[segment_indices[0], 'sig_array']
    
    # Sort indices
    sorted_indices = sorted(segment_indices)
    
    # Start with first signal
    reconstructed = data_df.loc[sorted_indices[0], 'sig_array'].copy()
    
    # Add subsequent signals, accounting for overlap
    for i in range(1, len(sorted_indices)):
        current_sig = data_df.loc[sorted_indices[i], 'sig_array']
        # Add only the non-overlapping portion (last 50 time steps)
        reconstructed = np.concatenate([reconstructed, current_sig[-step_size:]], axis=0)
    
    return reconstructed


def create_segment_dataframes(segments, data_df, label_encoder=None, use_predictions=False):
    """
    Create a DataFrame where each row represents one segment.
    """
    segment_data = []
    rest_classes = ['rest', 'deviceontable']
    if label_encoder is not None:
        rest_indices = [list(label_encoder.classes_).index(cls) for cls in rest_classes if cls in label_encoder.classes_]
    else:
        rest_indices = rest_classes  # When predictions are strings
    for i, segment in enumerate(segments):
        # Reconstruct full signal
        full_signal = reconstruct_segment_signal(segment['indices'], data_df)
        
        # Calculate segment statistics
        segment_length = len(segment['indices'])
        segment_duration = len(full_signal)  # Number of time steps
        
        # Determine the dominant exercise using probabilities if available
        if use_predictions and segment.get('probs') is not None and len(segment['probs']) > 0:
            # Get non-rest predictions and probabilities
            exercise_probs = []
            exercise_preds = []
            
            for j, pred in enumerate(segment['predictions']):
                if pred not in rest_indices:
                    exercise_probs.append(segment['probs'][j])
                    exercise_preds.append(pred)
            
            if len(exercise_probs) > 0:
                # Sum probabilities across all non-rest predictions
                total_prob_distribution = np.sum(exercise_probs, axis=0)
                
                # Find exercise with highest total probability
                most_common_pred = np.argmax(total_prob_distribution)
                dominant_prob = total_prob_distribution[most_common_pred]
                total_prob_sum = np.sum(total_prob_distribution)
                segment_confidence = dominant_prob / total_prob_sum if total_prob_sum > 0 else 0
                
                # Also calculate vote-based confidence for comparison
                pred_counter = Counter(exercise_preds)
                vote_based_pred, vote_count = pred_counter.most_common(1)[0]
                vote_confidence = vote_count / len(exercise_preds)
            else:
                # Fallback if no exercise predictions (only rest)
                pred_counter = Counter(segment['predictions'])
                most_common_pred, most_common_count = pred_counter.most_common(1)[0]
                segment_confidence = most_common_count / len(segment['predictions'])
                vote_confidence = segment_confidence
        else:
            # Use traditional vote-based approach
            pred_counter = Counter(segment['predictions'])
            most_common_pred, most_common_count = pred_counter.most_common(1)[0]
            segment_confidence = most_common_count / len(segment['predictions'])
            vote_confidence = segment_confidence
        
        # Get exercise name
        exercise_name = label_encoder.classes_[most_common_pred] if label_encoder is not None else most_common_pred
        
        # Get repetitions data (most common value in segment)
        if 'repetitions' in data_df.columns:
            segment_repetitions = data_df.loc[segment['indices'], 'repetitions'].values
            # Filter out any NaN values and get most common
            valid_reps = segment_repetitions[~pd.isna(segment_repetitions)]
            if len(valid_reps) > 0:
                rep_counter = Counter(valid_reps)
                most_common_reps = rep_counter.most_common(1)[0][0]
            else:
                most_common_reps = np.nan
        else:
            most_common_reps = np.nan
        
        segment_info = {
            'segment_id': i,
            'start_idx': segment['start_idx'],
            'end_idx': segment['end_idx'],
            'subject_id': segment['subject_id'],
            'exercise_name': exercise_name,
            'exercise_label': most_common_pred,
            'segment_length': segment_length,
            'signal_duration': segment_duration,
            'confidence': segment_confidence,
            'repetitions': most_common_reps,
            'sig_array': full_signal,
            'indices': segment['indices']
        }
        
        if use_predictions and segment.get('probs') is not None:
            # Add probability-based metrics
            if len(exercise_probs) > 0:
                segment_info['prob_confidence'] = segment_confidence
                segment_info['vote_confidence'] = vote_confidence
                segment_info['avg_class_prob'] = dominant_prob / len(exercise_probs)  # Average prob for dominant class
                
                # Store the full probability distribution for the segment
                normalized_total_probs = total_prob_distribution / total_prob_sum if total_prob_sum > 0 else total_prob_distribution
                segment_info['segment_prob_distribution'] = normalized_total_probs
            else:
                segment_info['prob_confidence'] = segment_confidence
                segment_info['vote_confidence'] = segment_confidence
        
        segment_data.append(segment_info)
    
    return pd.DataFrame(segment_data)


def calculate_segment_overlap(true_start, true_end, pred_start, pred_end):
    """
    Calculate overlap metrics between two segments.
    Returns overlap_length, iou, overlap_ratio_true, overlap_ratio_pred
    """
    # Calculate overlap
    overlap_start = max(true_start, pred_start)
    overlap_end = min(true_end, pred_end)
    overlap_length = max(0, overlap_end - overlap_start + 1)
    
    # Calculate segment lengths
    true_length = true_end - true_start + 1
    pred_length = pred_end - pred_start + 1
    
    # Calculate IoU (Intersection over Union)
    union_length = true_length + pred_length - overlap_length
    iou = overlap_length / union_length if union_length > 0 else 0
    
    # Calculate overlap ratios
    overlap_ratio_true = overlap_length / true_length if true_length > 0 else 0
    overlap_ratio_pred = overlap_length / pred_length if pred_length > 0 else 0
    
    return overlap_length, iou, overlap_ratio_true, overlap_ratio_pred


def fuzzy_match_segments(true_segments_df, pred_segments_df, 
                        min_overlap_ratio=0.3, min_iou=0.2, 
                        require_same_exercise=True, require_same_subject=True):
    """
    Find fuzzy matches between true and predicted segments.
    """
    
    matches = []
    
    for true_idx, true_row in true_segments_df.iterrows():
        true_start = true_row['start_idx']
        true_end = true_row['end_idx']
        true_exercise = true_row['exercise_name']
        true_subject = true_row['subject_id']
        
        best_matches = []  # Store all potential matches for this true segment
        
        for pred_idx, pred_row in pred_segments_df.iterrows():
            pred_start = pred_row['start_idx']
            pred_end = pred_row['end_idx']
            pred_exercise = pred_row['exercise_name']
            pred_subject = pred_row['subject_id']
            
            # Check basic requirements
            if require_same_subject and true_subject != pred_subject:
                continue
                
            if require_same_exercise and true_exercise != pred_exercise:
                continue
            
            # Calculate overlap metrics
            overlap_length, iou, overlap_ratio_true, overlap_ratio_pred = calculate_segment_overlap(
                true_start, true_end, pred_start, pred_end
            )
            
            # Check if this is a valid match
            min_overlap_ratio_check = max(overlap_ratio_true, overlap_ratio_pred) >= min_overlap_ratio
            iou_check = iou >= min_iou
            
            if min_overlap_ratio_check and iou_check:
                # Calculate a composite match score
                exercise_match_score = 1.0 if true_exercise == pred_exercise else 0.0
                subject_match_score = 1.0 if true_subject == pred_subject else 0.0
                
                # Weighted composite score
                match_score = (
                    0.4 * iou + 
                    0.3 * max(overlap_ratio_true, overlap_ratio_pred) +
                    0.2 * exercise_match_score +
                    0.1 * subject_match_score
                )
                
                match_info = {
                    'true_segment_id': true_row['segment_id'],
                    'pred_segment_id': pred_row['segment_id'],
                    'match_score': match_score,
                    'overlap_length': overlap_length,
                    'iou': iou,
                    'overlap_ratio_true': overlap_ratio_true,
                    'overlap_ratio_pred': overlap_ratio_pred,
                    'exercise_match': true_exercise == pred_exercise,
                    'subject_match': true_subject == pred_subject,
                    
                    # True segment data
                    'true_start_idx': true_start,
                    'true_end_idx': true_end,
                    'true_exercise_name': true_exercise,
                    'true_subject_id': true_subject,
                    'true_segment_length': true_row['segment_length'],
                    'true_signal_duration': true_row['signal_duration'],
                    'true_repetitions': true_row.get('repetitions', np.nan),
                    'true_sig_array': true_row['sig_array'],
                    
                    # Predicted segment data
                    'pred_start_idx': pred_start,
                    'pred_end_idx': pred_end,
                    'pred_exercise_name': pred_exercise,
                    'pred_subject_id': pred_subject,
                    'pred_segment_length': pred_row['segment_length'],
                    'pred_signal_duration': pred_row['signal_duration'],
                    'pred_confidence': pred_row['confidence'],
                    'pred_sig_array': pred_row['sig_array'],
                }
                
                # Add prediction-specific columns if they exist
                if 'avg_class_prob' in pred_row:
                    match_info['pred_avg_class_prob'] = pred_row['avg_class_prob']
                
                best_matches.append(match_info)
        
        # Sort matches by score and keep the best ones
        best_matches.sort(key=lambda x: x['match_score'], reverse=True)
        matches.extend(best_matches)
    
    return pd.DataFrame(matches)


# Repetition counting classes and functions
class ResearchBasedRepCounter:
    """
    Implementation based on research paper approach for repetition counting
    using accelerometer and gyroscope data.
    """
    
    def __init__(self, sampling_rate: float = 75):
        self.sampling_rate = sampling_rate
        
        # Exercise-specific timing constraints (in seconds)
        self.exercise_timing = {
            # Upper body exercises
            'bicepcurls': {'min_rep_time': 1.0, 'max_rep_time': 4.0},
            'dumbbellshoulderpress': {'min_rep_time': 1.0, 'max_rep_time': 4.0},
            'lateralshoulderraises': {'min_rep_time': 0.8, 'max_rep_time': 3.0},
            'tricepextensions': {'min_rep_time': 1.0, 'max_rep_time': 4.0},
            'dumbbellrows': {'min_rep_time': 1.0, 'max_rep_time': 4.0},
            'seatedbackfly': {'min_rep_time': 1.0, 'max_rep_time': 4.0},
            
            # Bodyweight exercises
            'pushups': {'min_rep_time': 0.8, 'max_rep_time': 3.0},
            'squats': {'min_rep_time': 1.5, 'max_rep_time': 4.0},
            'lunges': {'min_rep_time': 1.0, 'max_rep_time': 4.0},
            'dip': {'min_rep_time': 1.0, 'max_rep_time': 4.0},
            
            # Core exercises
            'situps': {'min_rep_time': 1.25, 'max_rep_time': 3.0},
            'crunch': {'min_rep_time': 0.8, 'max_rep_time': 3.0},
            'v-up': {'min_rep_time': 1.0, 'max_rep_time': 3.0},
            'butterflysit-up': {'min_rep_time': 1.5, 'max_rep_time': 3.5},
            'russiantwist': {'min_rep_time': 0.25, 'max_rep_time': 1.5},
            
            # High-intensity exercises
            'burpee': {'min_rep_time': 2.0, 'max_rep_time': 6.0},
            'kettlebellswing': {'min_rep_time': 1.0, 'max_rep_time': 3.5},
            'fastalternatingpunches': {'min_rep_time': 0.1, 'max_rep_time': 0.5},
            
            # Default for unknown exercises
            'default': {'min_rep_time': 0.8, 'max_rep_time': 4.0}
        }
    
    def get_exercise_timing(self, exercise_name: str) -> Tuple[float, float]:
        """Get timing constraints for specific exercise"""
        exercise_key = exercise_name.lower().replace(' ', '').replace('-', '')
        
        if exercise_key in self.exercise_timing:
            timing = self.exercise_timing[exercise_key]
        else:
            timing = self.exercise_timing['default']
            
        return timing['min_rep_time'], timing['max_rep_time']
    
    def preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Preprocess the 6D signal (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
        """
        if len(signal) < 10:
            return signal
        
        # Standardize each dimension
        scaler = StandardScaler()
        signal_standardized = scaler.fit_transform(signal)
        
        # Apply Savitzky-Golay filter (3rd degree polynomial)
        window_length = min(max(5, len(signal) // 10), 21)  # Adaptive window
        if window_length % 2 == 0:
            window_length += 1
            
        if window_length >= 5:  # Need at least 5 points for 3rd degree
            signal_smoothed = np.zeros_like(signal_standardized)
            for i in range(signal.shape[1]):
                try:
                    signal_smoothed[:, i] = savgol_filter(
                        signal_standardized[:, i], 
                        window_length, 
                        3  # 3rd degree polynomial
                    )
                except:
                    # Fallback to original if filtering fails
                    signal_smoothed[:, i] = signal_standardized[:, i]
        else:
            signal_smoothed = signal_standardized
            
        return signal_smoothed
    
    def find_stable_exercise_region(self, signal: np.ndarray) -> Tuple[int, int]:
        """
        Very conservative approach: only trim obvious large spikes at start/end.
        """
        if len(signal) < 50:
            return 0, len(signal)
        
        # Use simple magnitude for detection
        signal_magnitude = np.linalg.norm(signal, axis=1)
        
        # Only consider trimming if signal is long enough
        max_trim_each_side = int(0.15 * len(signal))  # Max 15%
        
        if max_trim_each_side < 3:
            return 0, len(signal)  # Too short to trim
        
        # Calculate overall signal characteristics
        overall_median = np.median(signal_magnitude)
        overall_mad = np.median(np.abs(signal_magnitude - overall_median))
        
        # Define extreme outlier threshold (very conservative)
        outlier_threshold = overall_median + 3 * overall_mad
        
        # Check if start needs trimming
        start_trim = 0
        start_segment = signal_magnitude[:max_trim_each_side]
        if np.any(start_segment > outlier_threshold):
            for i in range(len(start_segment)):
                if start_segment[i] <= outlier_threshold:
                    start_trim = i
                    break
        
        # Check if end needs trimming
        end_trim = 0
        end_segment = signal_magnitude[-max_trim_each_side:]
        if np.any(end_segment > outlier_threshold):
            for i in range(len(end_segment)-1, -1, -1):
                if end_segment[i] <= outlier_threshold:
                    end_trim = len(end_segment) - 1 - i
                    break
        
        # Apply very conservative limits
        final_start = min(start_trim, max_trim_each_side)
        final_end = max(len(signal) - end_trim, len(signal) - max_trim_each_side)
        
        return final_start, final_end
    
    def project_to_principal_component(self, signal: np.ndarray) -> np.ndarray:
        """
        Project the multidimensional signal onto its first principal component
        """
        if signal.shape[0] < 2:
            return np.linalg.norm(signal, axis=1)
        
        try:
            pca = PCA(n_components=1)
            signal_1d = pca.fit_transform(signal).flatten()
            self.last_pca_variance_ratio = pca.explained_variance_ratio_[0]
            return signal_1d
        except:
            return np.linalg.norm(signal, axis=1)
    
    def find_peaks(self, signal_1d: np.ndarray) -> List[int]:
        """Find peaks (local maxima) by comparing neighboring values"""
        peaks = []
        
        for i in range(1, len(signal_1d) - 1):
            if signal_1d[i] > signal_1d[i-1] and signal_1d[i] > signal_1d[i+1]:
                peaks.append(i)
        
        return peaks
    
    def filter_peaks_by_distance(self, peaks: List[int], signal_1d: np.ndarray, 
                                min_distance_samples: int) -> List[int]:
        """Filter peaks by minimum distance, keeping highest amplitude"""
        if not peaks:
            return []
        
        # Get peak amplitudes and sort by amplitude (descending)
        peak_amplitudes = [(i, signal_1d[i]) for i in peaks]
        peak_amplitudes.sort(key=lambda x: x[1], reverse=True)
        
        filtered_peaks = []
        
        for peak_idx, amplitude in peak_amplitudes:
            # Check if this peak is too close to any already accepted peak
            too_close = False
            for accepted_peak in filtered_peaks:
                if abs(peak_idx - accepted_peak) < min_distance_samples:
                    too_close = True
                    break
            
            if not too_close:
                filtered_peaks.append(peak_idx)
        
        return sorted(filtered_peaks)  # Return in temporal order
    
    def compute_autocorrelation_period(self, signal_1d: np.ndarray, peak_idx: int,
                                     min_lag_samples: int, max_lag_samples: int,
                                     window_radius: int = None) -> float:
        """Compute autocorrelation for a window centered at the peak"""
        if window_radius is None:
            window_radius = max_lag_samples
        
        # Define window around the peak
        start_idx = max(0, peak_idx - window_radius)
        end_idx = min(len(signal_1d), peak_idx + window_radius)
        window_signal = signal_1d[start_idx:end_idx]
        
        if len(window_signal) < min_lag_samples + 1:
            return max_lag_samples
        
        # Compute autocorrelation for different lags
        max_autocorr = -1
        best_period = max_lag_samples
        
        for lag in range(min_lag_samples, min(max_lag_samples + 1, len(window_signal))):
            if lag >= len(window_signal):
                break
                
            # Compute autocorrelation at this lag
            signal_part1 = window_signal[:-lag]
            signal_part2 = window_signal[lag:]
            
            if len(signal_part1) > 0 and len(signal_part2) > 0:
                # Normalize both parts
                part1_norm = (signal_part1 - np.mean(signal_part1)) / (np.std(signal_part1) + 1e-8)
                part2_norm = (signal_part2 - np.mean(signal_part2)) / (np.std(signal_part2) + 1e-8)
                
                # Compute correlation
                autocorr = np.mean(part1_norm * part2_norm)
                
                if autocorr > max_autocorr:
                    max_autocorr = autocorr
                    best_period = lag
        
        return best_period
    
    def filter_peaks_by_autocorrelation(self, peaks: List[int], signal_1d: np.ndarray,
                                       min_lag_samples: int, max_lag_samples: int) -> List[int]:
        """Remove peaks based on autocorrelation analysis"""
        if len(peaks) <= 1:
            return peaks
        
        filtered_peaks = []
        
        for i, peak_idx in enumerate(peaks):
            # Compute period for this peak
            period = self.compute_autocorrelation_period(
                signal_1d, peak_idx, min_lag_samples, max_lag_samples
            )
            
            # Check if there are other peaks within 0.75*period that have higher amplitude
            exclusion_distance = int(0.75 * period)
            peak_amplitude = signal_1d[peak_idx]
            
            should_keep = True
            for other_peak_idx in peaks:
                if other_peak_idx == peak_idx:
                    continue
                
                distance = abs(other_peak_idx - peak_idx)
                if distance <= exclusion_distance:
                    other_amplitude = signal_1d[other_peak_idx]
                    if other_amplitude > peak_amplitude:
                        should_keep = False
                        break
            
            if should_keep:
                filtered_peaks.append(peak_idx)
        
        return filtered_peaks
    
    def filter_peaks_by_amplitude(self, peaks: List[int], signal_1d: np.ndarray,
                                 percentile: int = 40) -> List[int]:
        """Final filtering: remove peaks with low amplitude"""
        if not peaks:
            return peaks
        
        # Get all peak amplitudes
        peak_amplitudes = [signal_1d[i] for i in peaks]
        
        # Compute threshold: half of the percentile value
        threshold = 0.5 * np.percentile(peak_amplitudes, percentile)
        
        # Filter peaks
        filtered_peaks = [peak_idx for peak_idx in peaks 
                         if signal_1d[peak_idx] >= threshold]
        
        return filtered_peaks
    
    def count_repetitions(self, signal: np.ndarray, exercise_name: str = "default",
                         debug: bool = False) -> int:
        """Main method implementing the research paper approach"""
        if len(signal) < 10:
            return 1
        
        debug_info = {}
        
        # Step 1: Preprocess signal
        processed_signal = self.preprocess_signal(signal)
        stable_start, stable_end = self.find_stable_exercise_region(processed_signal)
        
        debug_info['stable_region'] = {
            'start': stable_start,
            'end': stable_end,
            'original_length': len(signal),
            'stable_length': stable_end - stable_start,
            'excluded_start': 0,
            'excluded_end': 0
        }

        if stable_start > 0 or stable_end < len(processed_signal):
            processed_signal = processed_signal[stable_start:stable_end]
            debug_info['stable_region']['excluded_start'] = stable_start
            debug_info['stable_region']['excluded_end'] = len(signal) - stable_end
        
        # Step 2: Project to first principal component
        signal_1d = self.project_to_principal_component(processed_signal)
        debug_info['signal_1d'] = signal_1d
        debug_info['pca_variance_ratio'] = getattr(self, 'last_pca_variance_ratio', 0)
        
        # Step 3: Get exercise timing constraints
        min_rep_time, max_rep_time = self.get_exercise_timing(exercise_name)
        min_distance_samples = int(min_rep_time * self.sampling_rate)
        min_lag_samples = int(min_rep_time * self.sampling_rate)
        max_lag_samples = int(max_rep_time * self.sampling_rate)
        
        debug_info['timing'] = {
            'min_rep_time': min_rep_time,
            'max_rep_time': max_rep_time,
            'min_distance_samples': min_distance_samples
        }
        
        # Step 4: Find initial peaks
        initial_peaks = self.find_peaks(signal_1d)
        debug_info['initial_peaks'] = initial_peaks
        
        if not initial_peaks:
            return 1 if not debug else {'repetitions': 1, 'debug': debug_info}
        
        # Step 5: Filter peaks by minimum distance
        distance_filtered_peaks = self.filter_peaks_by_distance(
            initial_peaks, signal_1d, min_distance_samples
        )
        debug_info['distance_filtered_peaks'] = distance_filtered_peaks
        
        # Step 6: Filter peaks using autocorrelation
        autocorr_filtered_peaks = self.filter_peaks_by_autocorrelation(
            distance_filtered_peaks, signal_1d, min_lag_samples, max_lag_samples
        )
        debug_info['autocorr_filtered_peaks'] = autocorr_filtered_peaks
        
        # Step 7: Final amplitude filtering
        final_peaks = self.filter_peaks_by_amplitude(
            autocorr_filtered_peaks, signal_1d, percentile=40
        )
        debug_info['final_peaks'] = final_peaks
        
        rep_count = len(final_peaks)
        
        if debug:
            return {'repetitions': rep_count, 'debug': debug_info}
        
        return rep_count
    
    def predict_repetitions(self, signal: np.ndarray, exercise_name: str = "default") -> int:
        """Wrapper for main counting method to match interface"""
        return self.count_repetitions(signal, exercise_name)


def apply_research_counter_to_dataframe(df: pd.DataFrame, 
                                      signal_col: str = 'pred_sig_array',
                                      exercise_col: str = 'exercise',
                                      counter: Optional[ResearchBasedRepCounter] = None) -> pd.DataFrame:
    """Apply the research-based counter to a dataframe"""
    
    if counter is None:
        counter = ResearchBasedRepCounter()
    
    df = df.copy()
    predictions = []
    
    for idx, row in df.iterrows():
        try:
            signal = row[signal_col]
            exercise = row[exercise_col] if exercise_col in row else "default"
            pred_reps = counter.count_repetitions(signal, exercise)
            predictions.append(pred_reps)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            predictions.append(1)  # fallback
    
    df['predicted_repetitions'] = predictions
    return df


# Analysis functions
def analyze_by_exercise_group(df: pd.DataFrame, 
                             exercise_col: str = 'true_exercise_name', 
                             true_col: str = 'true_repetitions', 
                             pred_col: str = 'predicted_repetitions',
                             min_samples: int = 5) -> pd.DataFrame:
    """Comprehensive analysis of rep counting performance by exercise group"""
    
    # Calculate errors
    df = df.copy()
    df['error'] = df[pred_col] - df[true_col]
    df['abs_error'] = np.abs(df['error'])
    df['within_1'] = df['abs_error'] <= 1
    df['within_2'] = df['abs_error'] <= 2
    df['within_3'] = df['abs_error'] <= 3
    
    # Group by exercise
    grouped = df.groupby(exercise_col)
    
    results = []
    
    for exercise_name, group in grouped:
        if len(group) < min_samples:
            continue
            
        n_samples = len(group)
        true_reps = group[true_col].values
        pred_reps = group[pred_col].values
        errors = group['error'].values
        abs_errors = group['abs_error'].values
        
        # Basic statistics
        mean_true = np.mean(true_reps)
        mean_pred = np.mean(pred_reps)
        
        # Error metrics
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(errors**2))
        mean_error = np.mean(errors)  # bias
        std_error = np.std(errors)
        
        # Accuracy metrics
        acc_within_1 = np.mean(group['within_1'])
        acc_within_2 = np.mean(group['within_2'])
        acc_within_3 = np.mean(group['within_3'])
        perfect_accuracy = np.mean(abs_errors == 0)
        
        # Error distribution
        min_error = np.min(errors)
        max_error = np.max(errors)
        p25_error = np.percentile(errors, 25)
        p75_error = np.percentile(errors, 75)
        
        # Over/under counting analysis
        over_count_rate = np.mean(errors > 0)
        under_count_rate = np.mean(errors < 0)
        
        # Worst cases
        worst_overcount = np.max(errors) if len(errors) > 0 else 0
        worst_undercount = np.min(errors) if len(errors) > 0 else 0
        
        results.append({
            'true_exercise_name': exercise_name,
            'n_samples': n_samples,
            'mean_true_reps': round(mean_true, 1),
            'mean_pred_reps': round(mean_pred, 1),
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'mean_error': round(mean_error, 2),
            'std_error': round(std_error, 2),
            'accuracy_within_1': round(acc_within_1, 3),
            'accuracy_within_2': round(acc_within_2, 3),
            'accuracy_within_3': round(acc_within_3, 3),
            'perfect_accuracy': round(perfect_accuracy, 3),
            'over_count_rate': round(over_count_rate, 3),
            'under_count_rate': round(under_count_rate, 3),
            'min_error': min_error,
            'max_error': max_error,
            'p25_error': p25_error,
            'p75_error': p75_error,
            'worst_overcount': worst_overcount,
            'worst_undercount': worst_undercount,
        })
    
    results_df = pd.DataFrame(results)
    return results_df


def print_exercise_summary(analysis_df: pd.DataFrame, top_n: int = 10):
    """Print a human-readable summary of the analysis"""
    
    print("="*80)
    print("EXERCISE REP COUNTING PERFORMANCE ANALYSIS")
    print("="*80)
    
    total_samples = analysis_df['n_samples'].sum()
    overall_acc = np.average(analysis_df['accuracy_within_1'], weights=analysis_df['n_samples'])
    overall_mae = np.average(analysis_df['mae'], weights=analysis_df['n_samples'])
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Total samples: {total_samples}")
    print(f"  Overall accuracy (±1 rep): {overall_acc:.1%}")
    print(f"  Overall MAE: {overall_mae:.2f} reps")
    
    print(f"\nWORST PERFORMING EXERCISES (top {min(top_n, len(analysis_df))}):")
    worst = analysis_df.head(top_n)
    for _, row in worst.iterrows():
        print(f"  {row['true_exercise_name']:<20} | Acc: {row['accuracy_within_1']:.1%} | MAE: {row['mae']:.2f} | "
              f"Bias: {row['mean_error']:+.1f} | n={row['n_samples']}")
    
    print(f"\nBEST PERFORMING EXERCISES (top {min(top_n, len(analysis_df))}):")
    best = analysis_df.tail(top_n).iloc[::-1]
    for _, row in best.iterrows():
        print(f"  {row['true_exercise_name']:<20} | Acc: {row['accuracy_within_1']:.1%} | MAE: {row['mae']:.2f} | "
              f"Bias: {row['mean_error']:+.1f} | n={row['n_samples']}")


def plot_exercise_performance(analysis_df: pd.DataFrame, 
                            metric: str = 'accuracy_within_1',
                            figsize: tuple = (12, 8)):
    """Create visualization of performance by exercise"""
    
    plt.figure(figsize=figsize)
    
    if metric == 'accuracy_within_1':
        title = 'Accuracy Within ±1 Rep by Exercise'
        ylabel = 'Accuracy'
        format_func = lambda x: f'{x:.1%}'
    elif metric == 'mae':
        title = 'Mean Absolute Error by Exercise'
        ylabel = 'MAE (reps)'
        format_func = lambda x: f'{x:.2f}'
    elif metric == 'mean_error':
        title = 'Bias (Over/Under Counting) by Exercise'
        ylabel = 'Mean Error (reps)'
        format_func = lambda x: f'{x:+.1f}'
    else:
        title = f'{metric} by Exercise'
        ylabel = metric
        format_func = lambda x: f'{x:.2f}'
    
    # Sort for better visualization
    plot_df = analysis_df.sort_values(metric, ascending=True)
    
    # Color code based on performance
    colors = []
    for val in plot_df[metric]:
        if metric == 'accuracy_within_1':
            if val >= 0.8:
                colors.append('green')
            elif val >= 0.6:
                colors.append('orange') 
            else:
                colors.append('red')
        elif metric == 'mae':
            if val <= 1.0:
                colors.append('green')
            elif val <= 2.0:
                colors.append('orange')
            else:
                colors.append('red')
        else:
            colors.append('steelblue')
    
    bars = plt.barh(range(len(plot_df)), plot_df[metric], color=colors, alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, plot_df[metric])):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                format_func(val), va='center', fontsize=9)
    
    plt.yticks(range(len(plot_df)), plot_df['true_exercise_name'])
    plt.xlabel(ylabel)
    plt.title(title)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()