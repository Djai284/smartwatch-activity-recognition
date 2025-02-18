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



def load_mmfit(filepath = 'mm-fit'):

       raw_data = pd.DataFrame(columns = ['index', 'frame', 'timestamp', 'acc_X', 'acc_Y', 'acc_Z', 'gyr_X',
              'gyr_Y', 'gyr_Z', 'activity_name', 'subject_id'])

       for i in tqdm(range(21)):

              acc_data = load_modality(f'{filepath}/w{i:02}/w{i:02}_sw_l_acc.npy')
              gyr_data = load_modality(f'{filepath}/w{i:02}/w{i:02}_sw_l_gyr.npy')
              label_data = pd.read_csv(f'{filepath}/w{i:02}/w{i:02}_labels.csv', names = ['start_frame', 'end_frame', 'repititions', 'activity_name'])

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
    
    result_ids = []
    result_sig_arrays = []
    result_activity_names = []
    result_subject_ids = []
    result_datasets = []
    
    print("Creating dataset for model...")
    
    i = 0
    total_len = len(df)
    
    # Set up progress bar if requested
    pbar = tqdm(total=total_len) if show_progress else None
    
    while i < total_len:
        # Get initial values
        label = activity_array[i]
        user = subject_array[i]
        dataset = dataset_array[i]
        
        # Find the end of this segment (where activity or user changes)
        segment_end = i
        while segment_end < total_len and activity_array[segment_end] == label and subject_array[segment_end] == user:
            segment_end += 1
        
        # Process this segment
        j = i
        while j + dim <= segment_end:
            # Extract the chunk of sensor data
            arr = sensor_data[j:j+dim].copy()
            
            # Add to results
            result_ids.append(len(result_ids))
            result_sig_arrays.append(arr)
            result_activity_names.append(label)
            result_subject_ids.append(user)
            result_datasets.append(dataset)
            
            # Move to next position
            if between is None:
                j += dim  # Non-overlapping windows
            else:
                j += between  # Overlapping windows with specified step
        
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
        'dataset': result_datasets
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
        elif (df.iloc[i]['activity_name'] != df.iloc[i-1]['activity_name'] or 
              df.iloc[i]['subject_id'] != df.iloc[i-1]['subject_id']):
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
    df['subject_id'] = person_id

    # start at the begnining of the labels
    label_tracker = 0
    label = label_df.loc[label_tracker, 'activity_name']
    
    # loop througbn the dataframe
    for i, row in df.iterrows():
        
        # move to next label if frame is past current exercise
        if row['frame'] > label_df.loc[label_tracker, 'end_frame']:
            label_tracker += 1

            # move to next exercise
            if label_tracker < len(label_df):
                label = label_df.loc[label_tracker, 'activity_name']

            # break from loop if no more exercises
            else:
                df.loc[i:, 'activity_name'] = 'non-e'
                break
        
        # assign proper label
        if row['frame'] > label_df.loc[label_tracker, 'start_frame']:
            df.at[i, 'activity_name'] = label
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