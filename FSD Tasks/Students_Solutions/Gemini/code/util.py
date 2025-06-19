'''
Various utility methods.
'''

from typing import List, Tuple
import numpy as np

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

LABELS = [label_to_tuple(i) for i in range(12)]



def plot_drive_data(data: np.ndarray) -> None:   
    # Labels for the 6 sensor signals
    labels = ['ax', 'ay', 'az', 'rx', 'ry', 'rz']

    # Create figure and 2x3 subplots
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))

    # Loop over each axis and sensor data to plot
    for i, ax in enumerate(axs.flat):
        # Plot the sensor data (without time)
        ax.plot(data[:, i], color='blue')

        # Set subplot title and axis labels
        ax.set_title(labels[i])
        if i < 3:
            ax.set_ylabel('m/s²')
        else:
            ax.set_ylabel('°/s')

        # Remove top and right borders for cleaner look
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Add x-axis label for the bottom subplots
        if i >= 3:
            ax.set_xlabel('Time Step')

    # Adjust spaces between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()