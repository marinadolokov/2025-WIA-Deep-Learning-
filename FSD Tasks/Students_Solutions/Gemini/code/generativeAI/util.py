'''
Various utility methods.
'''

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch 
import torch.nn as nn


def label_to_tuple(label: int) -> Tuple[bool, bool, bool, bool]:
    '''
    Convert a class label to a 4d vector binary encoding the brake status.
    Input: class label (0-11).
    Output: (front_left, front_right, rear_left, rear_right), where True indicates an irregularity.
    '''
    match label:
        case 0: return (False, False, False, False)
        case 1: return (False, False, True, False)
        case 2: return (False, False, False, True)
        case 3: return (False, False, True, True)
        case 4: return (True, False, False, False)
        case 5: return (False, True, False, False)
        case 6: return (True, True, False, False)
        case 7: return (True, False, True, False)
        case 8: return (False, True, False, True)
        case 9: return (True, False, False, True)
        case 10: return (False, True, True, False)
        case 11: return (True, True, True, True)
    raise ValueError(f"Invalid label: {label}. Expected a value between 0 and 11.")

def tuple_to_label(tup: Tuple[bool, bool, bool, bool]) -> int:
    '''
    Convert a 4d tuple binary encoding the brake status to a class label.
    Input: (front_left, front_right, rear_left, rear_right), where True indicates an irregularity.
    Output: class label (0-11).
    '''
    match tup:
        case (False, False, False, False): return 0
        case (False, False, True, False): return 1
        case (False, False, False, True): return 2
        case (False, False, True, True): return 3
        case (True, False, False, False): return 4
        case (False, True, False, False): return 5
        case (True, True, False, False): return 6
        case (True, False, True, False): return 7
        case (False, True, False, True): return 8
        case (True, False, False, True): return 9
        case (False, True, True, False): return 10
        case (True, True, True, True): return 11
    raise ValueError(f"Invalid tuple: {tup}. Expected a tuple of 4 booleans.")


def plot_drive_data(data: np.ndarray) -> None:   
    labels = ['ax', 'ay', 'az', 'rx', 'ry', 'rz']

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
    for i, ax in enumerate(axs.flat):
        ax.plot(data[:, i], color='blue')

        ax.set_title(labels[i])
        if i < 3:
            ax.set_ylabel('m/s²')
        else:
            ax.set_ylabel('°/s')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if i >= 3:
            ax.set_xlabel('Time Step')

    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()


def package_fake_series(fake_data, fake_labels, df_train):
    """
    packages fake generated sensor_data series as a row in the DataFrame format
    used for training
    """
    # Get median values for all columns except 'label' and 'sensor_data'
    median_values = df_train.drop(columns=['label', 'sensor_data', 'model']).median()

    sensor_data = []
    label = []
    model = []
    deceleration_average = []
    velocity = []
    mass = []

    for d, l in zip(fake_data.detach(), fake_labels.detach()):
        sensor_data.append(d.numpy())
        label.append(tuple_to_label(tuple(l.bool().tolist())))
        model.append('Fake Car 9000')
        deceleration_average.append(median_values['deceleration_average'])
        velocity.append(median_values['velocity'])
        mass.append(median_values['mass'])

    return pd.DataFrame({
        'sensor_data' : sensor_data,
        'label' : label,
        'model' : model,
        'deceleration_average' : deceleration_average,
        'velocity' : velocity,
        'mass' : mass})



def create_dataset(df):
    sensor_data = df['sensor_data'].apply(lambda x: np.array(x, dtype=np.float32))
    X = np.stack(sensor_data.values)
    y = df['label'].values
    y = np.array([label_to_tuple(label) for label in y])

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    return dataset

