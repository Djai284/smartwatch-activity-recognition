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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class IMUVisualizer:
    """
    Handles visualization of IMU sensor data with enhanced features.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the visualizer with a DataFrame.
        
        Args:
            df: DataFrame containing IMU data with required columns
        """
        self.df = df.copy()  # Create a copy to avoid modifying original data
        self._validate_dataframe()
        self.default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        
    def _validate_dataframe(self) -> None:
        """Validate that the DataFrame has all required columns."""
        required_columns = [
            'timestamp', 'label',
            'acc_X', 'acc_Y', 'acc_Z',
            'gyr_X', 'gyr_Y', 'gyr_Z'
        ]
        
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    def get_activity_segments(self, activity: str) -> List[pd.DataFrame]:
        """
        Split activity data into separate continuous segments.
        
        Args:
            activity: Name of the activity to segment
            
        Returns:
            List of DataFrames, each containing a continuous segment
        """
        activity_data = self.df[self.df['label'] == activity].copy()
        
        if activity_data.empty:
            logger.warning(f"No data found for activity: {activity}")
            return []
        
        # Sort by timestamp to ensure proper segmentation
        activity_data = activity_data.sort_values('timestamp')
        
        # Identify breaks between segments (gaps > 0.1 seconds)
        activity_data['time_diff'] = activity_data['timestamp'].diff()
        segment_breaks = activity_data[activity_data['time_diff'] > 0.1].index
        
        # Split into segments
        segments = []
        if len(segment_breaks) == 0:
            segments = [activity_data]
        else:
            indices = np.split(activity_data.index, segment_breaks)
            segments = [activity_data.loc[idx].copy() for idx in indices if len(idx) > 0]
        
        # Add relative timestamps to each segment
        for segment in segments:
            if not segment.empty:
                segment['relative_time'] = segment['timestamp'] - segment['timestamp'].iloc[0]
            
        return [seg for seg in segments if len(seg) > 0]

    def plot_imu_data(self,
                     activity: str,
                     sensor_type: str = 'both',
                     time_window: Optional[float] = None,
                     instance_index: int = 0,
                     figsize: tuple = (15, 8),
                     show_statistics: bool = True) -> Optional[Tuple[plt.Figure, Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]]]:
        """
        Plot IMU data for a specific activity with enhanced features.
        
        Args:
            activity: Name of the activity to plot
            sensor_type: 'acc' for accelerometer, 'gyr' for gyroscope, or 'both' for both
            time_window: Number of seconds to plot (None for entire activity)
            instance_index: Which instance of the activity to plot
            figsize: Figure size for the plot
            show_statistics: Whether to show summary statistics in the plot
            
        Returns:
            Tuple of (figure, axes) if successful, None otherwise
        """
        segments = self.get_activity_segments(activity)
        
        if not segments:
            logger.warning(f"No segments found for activity: {activity}")
            return None
            
        if instance_index >= len(segments):
            logger.warning(f"Instance index {instance_index} out of range. Maximum available: {len(segments)-1}")
            return None
        
        # Get data for the specified instance
        instance_data = segments[instance_index]
        
        # Ensure data is sorted by time
        instance_data = instance_data.sort_values('relative_time')
        
        # Apply time window if specified
        if time_window is not None:
            instance_data = instance_data[instance_data['relative_time'] <= time_window].copy()
        
        # Create the plot
        if sensor_type == 'both':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
            axes = (ax1, ax2)
        else:
            fig, ax1 = plt.subplots(figsize=figsize)
            axes = ax1
        
        # Plot accelerometer data
        if sensor_type in ['acc', 'both']:
            self._plot_sensor_data(ax1, instance_data, 'acc', show_statistics)
        
        # Plot gyroscope data
        if sensor_type in ['gyr', 'both']:
            ax = ax2 if sensor_type == 'both' else ax1
            self._plot_sensor_data(ax, instance_data, 'gyr', show_statistics)
        
        plt.tight_layout()
        return fig, axes

    def _plot_sensor_data(self, ax: plt.Axes, data: pd.DataFrame, 
                         sensor_type: str, show_statistics: bool) -> None:
        """Helper method to plot sensor data with statistics."""
        sensor_map = {
            'acc': ('Accelerometer', 'Acceleration (m/s²)', ['acc_X', 'acc_Y', 'acc_Z']),
            'gyr': ('Gyroscope', 'Angular Velocity (rad/s)', ['gyr_X', 'gyr_Y', 'gyr_Z'])
        }
        
        title, ylabel, columns = sensor_map[sensor_type]
        activity = data['label'].iloc[0]
        
        # Plot the data
        for col, color in zip(columns, self.default_colors):
            ax.plot(data['relative_time'], data[col], 
                   label=col.split('_')[1], color=color)
        
        # Add statistics if requested
        if show_statistics:
            stats_text = self._generate_statistics(data[columns])
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_title(f'{title} Data - {activity}')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Ensure axis limits are appropriate
        ax.set_xlim(data['relative_time'].min(), data['relative_time'].max())
        y_range = max(abs(data[columns].min().min()), abs(data[columns].max().max()))
        ax.set_ylim(-y_range * 1.1, y_range * 1.1)

    def _generate_statistics(self, data: pd.DataFrame) -> str:
        """Generate summary statistics text."""
        stats = []
        for col in data.columns:
            axis = col.split('_')[1]
            stats.extend([
                f"{axis}-axis statistics:",
                f"  Mean: {data[col].mean():.2f}",
                f"  Std: {data[col].std():.2f}",
                f"  Max: {data[col].max():.2f}",
                f"  Min: {data[col].min():.2f}"
            ])
        return '\n'.join(stats)

    def get_activity_summary(self, activity: str = None) -> pd.DataFrame:
        """
        Get summary statistics for one or all activities.
        
        Args:
            activity: Specific activity to summarize, or None for all activities
            
        Returns:
            DataFrame containing activity statistics
        """
        if activity:
            data = self.df[self.df['label'] == activity]
        else:
            data = self.df
            
        summaries = []
        for name, group in data.groupby('label'):
            segments = self.get_activity_segments(name)
            
            # Skip if no valid segments
            if not segments:
                continue
                
            summary = {
                'label': name,
                'total_instances': len(segments),
                'total_samples': len(group),
                'avg_duration': np.mean([seg['relative_time'].max() for seg in segments]),
                'min_duration': np.min([seg['relative_time'].max() for seg in segments]),
                'max_duration': np.max([seg['relative_time'].max() for seg in segments])
            }
            
            # Add sensor statistics
            for sensor in ['acc', 'gyr']:
                for axis in ['X', 'Y', 'Z']:
                    col = f'{sensor}_{axis}'
                    summary.update({
                        f'{col}_mean': group[col].mean(),
                        f'{col}_std': group[col].std(),
                        f'{col}_max': group[col].max(),
                        f'{col}_min': group[col].min()
                    })
            
            summaries.append(summary)
            
        return pd.DataFrame(summaries)

    def list_available_activities(self) -> pd.DataFrame:
        """
        List all available activities and their instance counts.
        
        Returns:
            DataFrame containing activity information
        """
        activities = []
        for activity in self.df['label'].unique():
            segments = self.get_activity_segments(activity)
            if segments:  # Only include activities with valid segments
                durations = [seg['relative_time'].max() for seg in segments]
                activities.append({
                    'activity': activity,
                    'instances': len(segments),
                    'total_samples': len(self.df[self.df['label'] == activity]),
                    'avg_duration': np.mean(durations),
                    'min_duration': np.min(durations),
                    'max_duration': np.max(durations)
                })
        
        return pd.DataFrame(activities).set_index('activity')

