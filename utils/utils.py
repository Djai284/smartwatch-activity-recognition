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

# Set up logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IMUDataProcessor:
    """
    Handles processing of IMU data specific to right-arm worn sensors.
    Combines functionality from both original files into a single, coherent class.
    """
    
    EXPECTED_SAMPLE_RATE = 50  # Hz
    DEFAULT_EXCLUDED_ACTIVITIES = [
        'Device on Table',
        'Non-Exercise',
        'Tap Left Device',
        'Tap Right Device',
        'Arm Band Adjustment',
        'Initial Activity',
        'Invalid',
        'Note',
        'Unlisted Exercise'
    ]
    
    @staticmethod
    def safe_extract_value(array_like, dtype=None):
        """
        Safely extract a scalar value from a numpy array.
        """
        try:
            if hasattr(array_like, 'item'):
                value = array_like.item()
            else:
                value = array_like
                
            if dtype is not None:
                value = dtype(value)
            return value
        except Exception as e:
            logger.error(f"Error extracting value: {str(e)}")
            return None

    @staticmethod
    def process_imu_data(instance: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process IMU data from the structured array format.
        """
        try:
            # Extract the IMU data fields directly from instance
            imu_struct = instance['data'][0,0][0]
            
            return {
                'accelerometer': imu_struct['accelDataMatrix'][0],
                'gyroscope': imu_struct['gyroDataMatrix'][0],
                'slave_accelerometer': imu_struct['slaveAccelDataMatrix'][0],
                'slave_gyroscope': imu_struct['slaveGyroDataMatrix'][0]
            }
        except Exception as e:
            logger.error(f"Error processing IMU data: {str(e)}")
            return None

class MatlabDataProcessor:
    """
    A unified class to handle loading, processing, and analyzing MATLAB exercise data.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.mat_contents = None
        self.imu_processor = IMUDataProcessor()
        self.activities = None
        self.processed_df = None
    
    def load_file(self) -> None:
        """
        Load the MATLAB file using scipy.io
        """
        try:
            self.mat_contents = scipy.io.loadmat(self.file_path)
            logger.info("File loaded successfully")
            
            # Load activities list if available
            if 'exerciseConstants' in self.mat_contents:
                self.activities = [act[0] for act in self.mat_contents['exerciseConstants'][0,0]['activities'][0]]
                logger.info(f"Loaded {len(self.activities)} activities")
            else:
                logger.warning("No exercise constants found in file")
            
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            raise

    def process_instance(self, instance: np.ndarray, row: int, col: int) -> Optional[Dict[str, Any]]:
        """
        Process a single exercise instance from the subject_data array.
        """
        try:
            # Skip empty instances
            if instance.size == 0:
                return None
            
            logger.debug(f"Processing instance at [{row},{col}]")
            
            # Basic instance information
            instance_data = {
                'matrix_row': row,
                'matrix_col': col
            }
            
            # Try to extract subject ID - field name might vary
            for field in ['subjectID', 'subjectIndex']:
                if field in instance.dtype.names:
                    subject_id = self.imu_processor.safe_extract_value(instance[field][0,0], dtype=int)
                    if subject_id is not None:
                        instance_data['subject_id'] = subject_id
                        break
            
            # Extract activity name
            if 'activityName' in instance.dtype.names:
                activity_name = self.imu_processor.safe_extract_value(instance['activityName'][0,0], dtype=str)
                if activity_name:
                    instance_data['activity_name'] = activity_name
            
            # Optional fields
            if 'activityReps' in instance.dtype.names and instance['activityReps'][0,0].size > 0:
                instance_data['activity_reps'] = self.imu_processor.safe_extract_value(instance['activityReps'][0,0], dtype=int)
            
            if 'activityVideoStartTimeSeconds' in instance.dtype.names and instance['activityVideoStartTimeSeconds'][0,0].size > 0:
                instance_data['start_time'] = self.imu_processor.safe_extract_value(instance['activityVideoStartTimeSeconds'][0,0], dtype=float)
            
            # Process IMU data
            imu_data = self.imu_processor.process_imu_data(instance)
            if imu_data:
                instance_data['imu_data'] = imu_data
                logger.debug(f"Successfully processed IMU data for instance [{row},{col}]")
            else:
                logger.warning(f"No IMU data processed for instance [{row},{col}]")
                return None
            
            return instance_data
            
        except Exception as e:
            logger.error(f"Error processing instance [{row},{col}]: {str(e)}")
            return None

    def convert_to_timeseries(self, instance_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Convert a processed instance into a time-series DataFrame.
        """
        try:
            imu_data = instance_data['imu_data']
            accel_data = imu_data['accelerometer']
            gyro_data = imu_data['gyroscope']
            
            n_samples = accel_data.shape[0]
            
            df_data = {
                'timestamp': accel_data[:, 0].astype(float),
                'acc_X': accel_data[:, 1].astype(float),
                'acc_Y': accel_data[:, 2].astype(float),
                'acc_Z': accel_data[:, 3].astype(float),
                'gyr_X': gyro_data[:, 1].astype(float),
                'gyr_Y': gyro_data[:, 2].astype(float),
                'gyr_Z': gyro_data[:, 3].astype(float),
                'matrix_row': [instance_data['matrix_row']] * n_samples,
                'matrix_col': [instance_data['matrix_col']] * n_samples
            }
            
            # Add required metadata if available
            for field in ['activity_name', 'subject_id']:
                if field in instance_data:
                    df_data[field] = [instance_data[field]] * n_samples
            
            # Add optional metadata
            for field in ['activity_reps', 'start_time']:
                if field in instance_data:
                    df_data[field] = [instance_data[field]] * n_samples
            
            return pd.DataFrame(df_data)
            
        except Exception as e:
            logger.error(f"Error converting to timeseries: {str(e)}")
            return None

    def process_all_data(self, convert_to_df: bool = True) -> Any:
        """
        Process all valid exercise instances in the subject_data array.
        """
        try:
            if 'subject_data' not in self.mat_contents:
                raise ValueError("No subject_data found in MATLAB file")
                
            subject_data = self.mat_contents['subject_data']
            logger.info(f"Processing data matrix of shape {subject_data.shape}")
            
            processed_instances = []
            all_dfs = []
            
            total_instances = 0
            valid_instances = 0
            
            for i in range(subject_data.shape[0]):
                for j in range(subject_data.shape[1]):
                    cell_data = subject_data[i, j]
                    total_instances += 1
                    
                    if cell_data.size > 0:
                        instance = self.process_instance(cell_data, i, j)
                        if instance:
                            valid_instances += 1
                            
                            if convert_to_df:
                                df = self.convert_to_timeseries(instance)
                                if df is not None:
                                    all_dfs.append(df)
                            else:
                                processed_instances.append(instance)
            
            logger.info(f"Processed {valid_instances} valid instances out of {total_instances} total cells")
            
            if convert_to_df:
                if not all_dfs:
                    logger.error("No valid DataFrames were created during processing")
                    raise ValueError("No valid data was processed")
                    
                final_df = pd.concat(all_dfs, ignore_index=True)
                final_df = final_df.sort_values(['subject_id', 'timestamp']).reset_index(drop=True)
                self.processed_df = final_df
                logger.info(f"Created final DataFrame with {len(final_df)} rows")
                return final_df
            else:
                return processed_instances
                
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise

    def get_dataset_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics for the processed dataset.
        """
        if self.processed_df is None:
            raise ValueError("No processed data available. Run process_all_data first.")
            
        df = self.processed_df
        summary = {
            'total_samples': len(df),
            'total_subjects': df['subject_id'].nunique(),
            'total_activities': df['activity_name'].nunique(),
            'samples_per_subject': df.groupby('subject_id').size().agg(['min', 'max', 'mean', 'median']).to_dict(),
            'activities_per_subject': df.groupby('subject_id')['activity_name'].nunique().agg(['min', 'max', 'mean', 'median']).to_dict(),
            'top_activities': df.groupby('activity_name').size().sort_values(ascending=False).head(10).to_dict()
        }
        
        if 'activity_reps' in df.columns:
            summary['reps_by_activity'] = df.groupby('activity_name')['activity_reps'].first().to_dict()
        
        return summary

    def filter_activities(self, exclude_activities: List[str] = None) -> pd.DataFrame:
        """
        Filter out specified activities from the processed DataFrame.
        """
        if self.processed_df is None:
            raise ValueError("No processed data available. Run process_all_data first.")
            
        if exclude_activities is None:
            exclude_activities = self.imu_processor.DEFAULT_EXCLUDED_ACTIVITIES
        
        filtered_df = self.processed_df[~self.processed_df['activity_name'].isin(exclude_activities)].reset_index(drop=True)
        return filtered_df

def process_matlab_data(file_path: str, convert_to_df: bool = True) -> Tuple[Any, List[str]]:
    """
    Convenience function to load and process the MATLAB file.
    """
    processor = MatlabDataProcessor(file_path)
    
    try:
        processor.load_file()
        processed_data = processor.process_all_data(convert_to_df=convert_to_df)
        return processed_data, processor.activities
    
    except Exception as e:
        logger.error(f"Error in data loading and processing: {str(e)}")
        raise

def load_mmfit():

       raw_data = pd.DataFrame(columns = ['index', 'frame', 'timestamp', 'acc_X', 'acc_Y', 'acc_Z', 'gyr_X',
              'gyr_Y', 'gyr_Z', 'label', 'user_id'])

       for i in tqdm(range(21)):

              acc_data = load_modality(f'mm-fit/w{i:02}/w{i:02}_sw_l_acc.npy')
              gyr_data = load_modality(f'mm-fit/w{i:02}/w{i:02}_sw_l_gyr.npy')
              label_data = pd.read_csv(f'mm-fit/w{i:02}/w{i:02}_labels.csv', names = ['start_frame', 'end_frame', 'repititions', 'label'])

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
        datapath = "/Users/jacobgottesman/Public/DS 4440/smartwatch-activity-recognition/crossfit_data/constrained_workout/preprocessed_numpy_data/np_exercise_data"

    if not info_path:
        info_path = 'participant_ex_code_map.txt'

    with open(info_path, 'rb') as f:
        part_info = json.load(f)

    df = pd.DataFrame(columns = ['acc_X', 'acc_Y', 'acc_Z', 'gyr_X', 'gyr_Y', 'gyr_Z', 'user_id', 'label'])

    for root, dirs, files in os.walk(datapath):

        # loading different exercise data
        for dir in tqdm(dirs):

            # loop through the files
            for root, dirs1, files1 in os.walk(datapath+'/'+dir):

                # loop through all the files
                for file in files1:


                    data = np.load(f'{datapath}/{dir}/{file}')
                    temp_df = pd.DataFrame(data[:6, :].T, columns = ['acc_X', 'acc_Y', 'acc_Z', 'gyr_X', 'gyr_Y', 'gyr_Z'])
                    temp_df['label'] = dir.lower().replace(' ', '')
                    temp_df['user_id'] = find_key_by_value(part_info, file)

                    df = pd.concat([df, temp_df], axis = 0)

    return df


def create_examples(df, seconds=10):
    ex_df = pd.DataFrame(columns=['id', 'sig_array', 'label', 'user_id'])
    label = df.loc[0, 'label']
    user = df.loc[0, 'user_id']
    counter = 0
    arr = []

    print("creating dataset for model")
    for i, row in tqdm(df.iterrows(), total= len(df)):
        if row['label'] != label or row['user_id'] !=  user:
            label = row['label']
            user = row['user_id']
            counter = 0
            arr = []

        arr.append(row[['gyr_X', 'gyr_Y', 'gyr_Z', 'acc_X', 'acc_Y', 'acc_Z']].values)
        counter += 1  # Increment counter

        if counter >= seconds * 100:
            ex_df.loc[len(ex_df)] = [len(ex_df), np.array(arr), label, user]
            counter = 0
            arr = []

    return ex_df


def create_df(acc_data, gyr_data, label_df, person_id = 0):

    # create dataframes for acceleromters and gyrosccopes
    acc_df = pd.DataFrame(acc_data, columns=['frame', 'timestamp', 'acc_X', 'acc_Y', 'acc_Z']).reset_index()
    gyr_df = pd.DataFrame(gyr_data, columns=['frame', 'timestamp', 'gyr_X', 'gyr_Y', 'gyr_Z']).reset_index()

    # combine the two dataframes
    df = pd.merge(acc_df, gyr_df, how = 'inner', on = ['index', 'frame', 'timestamp'])
    df['label'] = " "
    df['user_id'] = person_id

    # start at the begnining of the labels
    label_tracker = 0
    label = label_df.loc[label_tracker, 'label']
    
    # loop througbn the dataframe
    for i, row in df.iterrows():
        
        # move to next label if frame is past current exercise
        if row['frame'] > label_df.loc[label_tracker, 'end_frame']:
            label_tracker += 1

            # move to next exercise
            if label_tracker < len(label_df):
                label = label_df.loc[label_tracker, 'label']

            # break from loop if no more exercises
            else:
                df.loc[i:, 'label'] = 'non-e'
                break
        
        # assign proper label
        if row['frame'] > label_df.loc[label_tracker, 'start_frame']:
            df.at[i, 'label'] = label
        else:
            df.at[i, 'label'] = "non-e"

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