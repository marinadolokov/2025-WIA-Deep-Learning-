import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_pickle('FSDTask/2025_Data/test.pickle')

# Choice of Dataset (max 1001)
n_data = 20

def plot_drive_data(data) -> None:   

    label = data['label']
    model = data['model']
    mass = data['mass']
    sensor_data = data['sensor_data']

    legend = str(model) + ' with mass: ' + str(mass) + ' and brake status: ' + str(label)
    
    # Labels for the 6 sensor signals
    labels = ['ax', 'ay', 'az', 'rx', 'ry', 'rz']

    # Create figure and 2x3 subplots
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))

    # Loop over each axis and sensor data to plot
    for i, ax in enumerate(axs.flat):
        # Plot the sensor data (without time)
        ax.plot(sensor_data[:, i], color='blue')

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

    #print modell and brake status
    fig.suptitle(legend)
    # Adjust spaces between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()


plot_drive_data(df.iloc[n_data])

