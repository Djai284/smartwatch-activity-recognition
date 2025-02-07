from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import scipy.io
import warnings
from typing import Dict, Tuple, Any, Optional, List
import logging
from scipy import signal

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plot_sensor(data, data_type, title='', filepath=None, size=(15, 6), overlay=True):
    if data.shape[-1] == 5:
        data = data[:, (0, 2, 3, 4)]
    else:
        data = data[:, (0, 2)]

    data_t = data[:, 0] - data[0, 0]

    if data_type == 'acc':
        text = 'Accelerometer'
        ylabel = 'm/s^2'
    elif data_type == 'gyr':
        text = 'Gyroscope'
        ylabel = 'radians/s'
    elif data_type == 'mag':
        text = 'Magnetometer'
        ylabel = 'µT'
    elif data_type == 'hr':
        text = 'Heart Rate'
        ylabel = 'HR/min'
        overlay = True
    else:
        raise Exception('The data_type argument, {}, is not valid. Must be acc, gyr, mag, or hr'.format(data_type))

    if overlay:
        fig, ax = plt.subplots(1, 1, figsize=size, sharex=True)
        if data_type == 'hr':
            ax.plot(data_t, data[:, 1], 'r-')
        else:
            ax.plot(data_t, data[:, 1], 'r-', label='X')
            ax.plot(data_t, data[:, 2], 'g-', label='Y')
            ax.plot(data_t, data[:, 3], 'b-', label='Z')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Frame')
        if data_type != 'hr':
            ax.legend()
    else:
        fig, ax = plt.subplots(3, 1, figsize=size, sharex=True)
        ax[0].plot(data_t, data[:, 1], 'r-', label='X')
        ax[1].plot(data_t, data[:, 2], 'g-', label='Y')
        ax[2].plot(data_t, data[:, 3], 'b-', label='Z')
        ax[0].title.set_text(text + ' X-axis')
        ax[0].set_ylabel(ylabel)
        ax[1].title.set_text(text + ' Y-axis')
        ax[1].set_ylabel(ylabel)
        ax[2].title.set_text(text + ' Z-axis')
        ax[2].set_ylabel(ylabel)
        ax[2].set_xlabel('Frame')

    fig.suptitle(title, size=16)
    if filepath is not None:
        fig.savefig(filepath, bbox_inches='tight')
    plt.show()


def plot_joints(data, dim, joints):
    fig, ax = plt.subplots(len(joints) * dim, 1, figsize=(15, len(joints) * 2 * dim), sharex=True)
    fig.tight_layout()
    for i, joint in enumerate(joints):
        ax[i * dim].title.set_text('Joint: ' + str(joint) + ' X')
        ax[i * dim].plot(data[0, :, 0], data[0, :, joint + 1], 'r-', label='X')
        ax[i * dim].set_ylabel('Units')

        ax[i * dim + 1].title.set_text('Joint: ' + str(joint) + ' Y')
        ax[i * dim + 1].plot(data[0, :, 0], data[1, :, joint + 1], 'b-', label='Y')
        ax[i * dim + 1].set_ylabel('Units')

        if dim == 3:
            ax[i * dim + 2].title.set_text('Joint: ' + str(joint) + ' Z')
            ax[i * dim + 2].plot(data[0, :, 0], data[2, :, joint + 1], 'g-', label='Z')
            ax[i * dim + 2].set_ylabel('Units')

    ax[i * dim + (dim - 1)].set_xlabel('Frame')
    plt.show()


def plot_2d_pose(pose, figsize=(8, 8)):
    """
    Visualize a 2D skeleton.
    :param pose: numpy array (2 x 18) with x, y coordinates with COCO keypoint format.
    :param figsize: Figure size.
    :return: None.
    """

    fig, ax = plt.subplots(figsize=figsize)
    for joint in range(pose.shape[1]):
        ax.plot(pose[0, joint], pose[1, joint], 'r.', markersize=10)
        
    limbs = [(0, 1), (0, 14), (0, 15), (14, 16), (15, 17), (1, 2), (2, 3), (3, 4),
             (1, 5), (5, 6), (6, 7), (1, 8), (1, 11), (8, 9), (9, 10), (11, 12), (12, 13)]
    for limb in limbs:
        joint1_x, joint1_y = pose[0, limb[0]], pose[1, limb[0]]
        joint2_x, joint2_y = pose[0, limb[1]], pose[1, limb[1]]
        plt.plot([joint1_x, joint2_x], [joint1_y, joint2_y], 'k-')
    
    ax.set_xlabel('X', size=14)
    ax.set_ylabel('Y', rotation=0, size=14)
    radius = 300
    ax.set_xlim((np.mean(pose[0, :]) - radius, np.mean(pose[0, :]) + radius))
    ax.set_ylim((np.mean(pose[1, :]) - radius, np.mean(pose[1, :]) + radius))
    plt.gca().invert_yaxis()
    plt.title('2D Pose Estimate', size=14)
    plt.show()

    
def plot_3d_pose(pose, elev=0, azim=0, figsize=(8, 8)):
    """
    Visualize a 3D skeleton.
    :param pose: numpy array (3 x 17) with x, y, z coordinates with COCO keypoint format.
    :param elev: Elevation angle in the z plane.
    :param azim: Azimuth angle in the x, y plane.
    :param figsize: Figure size.
    :return: None
    """
    pose = pose.flatten(order='F')
    vals = np.reshape(pose, (17, -1))

    fig = plt.figure(figsize=figsize)
    ax = Axes3D(fig)
    ax.view_init(elev, azim)

    limbs = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
             (12, 13), (8, 14), (14, 15), (15, 16)]
    left_right_limb = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    for i, limb in enumerate(limbs):
        x, y, z = [np.array([vals[limb[0], j], vals[limb[1], j]]) for j in range(3)]
        if left_right_limb[i] == 0:
            cc = 'blue'
        elif left_right_limb[i] == 1:
            cc = 'red'
        else:
            cc = 'black'
        ax.plot(x, y, z, marker='o', markersize=2, lw=1, c=cc)

    radius = 650
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-radius + xroot, radius + xroot])
    ax.set_zlim3d([-radius + zroot, radius + zroot])
    ax.set_ylim3d([-radius + yroot, radius + yroot])

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    white = (1.0, 1.0, 0.1, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    ax.w_zaxis.set_pane_color(white)

def plot_sensor_label(data, data_type,label, title='', filepath=None, size=(15, 6), overlay=True, seconds = 10):


    data = data[(data.index > min(data[data['label'] == label].index)) & (data.index < 100*seconds+min(data[data['label'] == label].index))]

    data_t = range(len(data.index))

    if data_type == 'acc':
        text = 'Accelerometer'
        ylabel = 'm/s^2'

    elif data_type == 'gyr':
        text = 'Gyroscope'
        ylabel = 'radians/s'

    else:
        raise Exception('The data_type argument, {}, is not valid. Must be acc, gyr, mag, or hr'.format(data_type))

    fig, ax = plt.subplots(1, 1, figsize=size, sharex=True)

    ax.plot(data_t, data.loc[:, f'{data_type}_X'], 'r-', label='X')
    ax.plot(data_t, data.loc[:, f'{data_type}_Y'], 'g-', label='Y')
    ax.plot(data_t, data.loc[:, f'{data_type}_Z'], 'b-', label='Z')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('reading')
    plt.legend()

    fig.suptitle(title, size=16)
    if filepath is not None:
        fig.savefig(filepath, bbox_inches='tight')
    plt.show()

def plot_imu_data(df: pd.DataFrame,
                activity: str,
                sensor_type: str = 'both',
                time_window: Optional[float] = None,
                instance_index: int = 0,
                figsize: tuple = (15, 8)) -> None:
    # """
    # Plot IMU data for a specific activity.
    
    # Args:
    #     df: DataFrame containing IMU data
    #     activity: Name of the activity to plot
    #     sensor_type: 'acc' for accelerometer, 'gyr' for gyroscope, or 'both' for both
    #     time_window: Number of seconds to plot (None for entire activity)
    #     instance_index: Which instance of the activity to plot (if multiple exist)
    #     figsize: Figure size for the plot
    # """
    # Filter data for the specified activity
    activity_data = df[df['label'] == activity].copy()
    
    if activity_data.empty:
        print(f"No data found for activity: {activity}")
        return
    
    # Get unique instances based on continuous timestamps
    activity_data['time_diff'] = activity_data['timestamp'].diff()
    instance_breaks = activity_data[activity_data['time_diff'] > 1].index
    instance_indices = np.split(activity_data.index, instance_breaks)
    
    if instance_index >= len(instance_indices):
        print(f"Instance index {instance_index} is out of range. Maximum available: {len(instance_indices)-1}")
        return
    
    # Get data for the specified instance
    instance_data = activity_data.loc[instance_indices[instance_index]]
    
    # Adjust timestamps to start from 0
    instance_data['relative_time'] = instance_data['timestamp'] - instance_data['timestamp'].iloc[0]
    
    # Apply time window if specified
    if time_window is not None:
        instance_data = instance_data[instance_data['relative_time'] <= time_window]
    
    # Create the plot
    if sensor_type == 'both':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot accelerometer data
    if sensor_type in ['acc', 'both']:
        ax1.plot(instance_data['relative_time'], instance_data['acc_X'], label='X', color='r')
        ax1.plot(instance_data['relative_time'], instance_data['acc_Y'], label='Y', color='g')
        ax1.plot(instance_data['relative_time'], instance_data['acc_Z'], label='Z', color='b')
        ax1.set_title(f'Accelerometer Data - {activity}')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Acceleration')
        ax1.grid(True)
        ax1.legend()
    
    # Plot gyroscope data
    if sensor_type in ['gyr', 'both']:
        ax = ax2 if sensor_type == 'both' else ax1
        ax.plot(instance_data['relative_time'], instance_data['gyr_X'], label='X', color='r')
        ax.plot(instance_data['relative_time'], instance_data['gyr_Y'], label='Y', color='g')
        ax.plot(instance_data['relative_time'], instance_data['gyr_Z'], label='Z', color='b')
        ax.set_title(f'Gyroscope Data - {activity}')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Angular Velocity')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.show()