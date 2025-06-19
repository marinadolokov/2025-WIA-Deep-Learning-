# import libraries
from comet_ml import Optimizer
import tensorflow as tf
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TerminateOnNaN

# import own files
from cnn_factory import build_cnn_from_experiment, generate_cnn_config
from fcn_factory import build_fcn_from_experiment, generate_fcn_config
from lstm_factory import build_lstm_model_from_experiment, generate_lstm_config
from utils import save_model_and_experiment, get_optimizer, TimeLimitCallback
from comet_user_management import set_comet_user_and_project_name, get_api_key, get_project_name

# constant parameters
RANDOM_SEED = 42                # random seed used in variations
NUM_RANDOM_SPLITS = 5           # number of random splits
EPOCHS = 100                    # epochs 
PATIENCE = 2                    # patience for early stopping
INPUT_SHAPE = (128, 6)          # input shape of training data
OUTPUT_UNITS = 12               # number of labels 
OUTPUT_ACTIVATION='softmax'     # activation used in output layer

set_comet_user_and_project_name("Martin", "Test")


# load train and test data
df = pd.read_pickle('train.pickle')
df_test = pd.read_pickle('test.pickle')

# extract data and labels for training and testing
all_train_data = np.stack(df.sensor_data.to_numpy())
all_train_labels = df.label.to_numpy()
test_data = np.stack(df_test.sensor_data.to_numpy())
test_labels = df_test.label.to_numpy()

# random split the train data into train and validation (0.2)
train_data      = [None] * NUM_RANDOM_SPLITS
train_labels    = [None] * NUM_RANDOM_SPLITS
valid_data      = [None] * NUM_RANDOM_SPLITS
valid_labels    = [None] * NUM_RANDOM_SPLITS

for i in range(NUM_RANDOM_SPLITS):
    train_data[i], valid_data[i], train_labels[i], valid_labels[i] =train_test_split(all_train_data, all_train_labels, test_size=0.2, random_state=RANDOM_SEED+i, shuffle=True)


def train(model,
          experiment,
          epochs: int, 
          batch_size: int, 
          optimizer: str,
          learning_rate: float, 
          num_data_set: int):
    """
    Trains a given model.
    mode: given model
    experiment: current comet_ml experiment
    epochs: number of epochs
    batch_size: size of batches
    optimizer: optimizer used in model compilation
    learnig_rate: learning rate for optimizer
    num_data_set: number that specifies which random split is trained
    """
    # compile model
    optimizer = get_optimizer(optimizer, learning_rate)
    loss=tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=PATIENCE,
                                                      restore_best_weights=True)
    time_limit = TimeLimitCallback(
        max_time=600,  # 10 minutes total
        max_time_per_epoch={0: 100, 1: 50, 8: 10},
        comet_experiment=experiment
    )

    # train the model
    model.fit(train_data[num_data_set], train_labels[num_data_set], batch_size, epochs,
              validation_data=(valid_data[num_data_set], valid_labels[num_data_set]),
              callbacks = [early_stopping, time_limit, TerminateOnNaN()])

    # evaluate the model on validation data
    val_loss, val_accuracy = model.evaluate(valid_data[num_data_set], valid_labels[num_data_set])

    # evaluate the model on test data
    loss, accuracy = model.evaluate(test_data, test_labels)

    return loss, accuracy, val_loss, val_accuracy



# instantiate optimizer
config = generate_lstm_config(max_num_lstm_layers=4,
                              max_num_lstm_units=64,
                              max_num_dense_layers=3,
                              random_seed=RANDOM_SEED,
                              metric_to_be_minimized="avg_val_loss",
                              resume_with_optimizer_id=None)

opt = Optimizer(config,                                
                api_key=get_api_key(),  # Do NOT hardcode the api key here (anymore)!
                project_name=get_project_name())

# run different configurations of hyperparameters
for experiment in opt.get_experiments():
    test_loss     = np.zeros(NUM_RANDOM_SPLITS)
    test_accuracy = np.zeros(NUM_RANDOM_SPLITS)
    val_loss = np.zeros(NUM_RANDOM_SPLITS)
    val_accuracy = np.zeros(NUM_RANDOM_SPLITS)

    # parameters used in every model
    learning_rate = experiment.get_parameter("learning_rate")
    batch_size = experiment.get_parameter("batch_size")
    optimizer = experiment.get_parameter("optimizer")

    # train model on every split of data
    for i in range(NUM_RANDOM_SPLITS):
        model = build_lstm_model_from_experiment(experiment=experiment,
                                                     input_shape=INPUT_SHAPE,
                                                     output_units=OUTPUT_UNITS,
                                                     output_activation=OUTPUT_ACTIVATION)
        test_loss[i], test_accuracy[i], val_loss[i], val_accuracy[i] = train(model, experiment,
                                                                             EPOCHS, batch_size, optimizer,
                                                                             learning_rate, i)

        experiment.log_metric("test_loss", test_loss[i])
        experiment.log_metric("test_accuracy", test_accuracy[i])

    avg_test_loss = np.average(test_loss)
    avg_test_accuracy = np.average(test_accuracy)

    avg_val_loss = np.average(val_loss)
    avg_val_accuracy = np.average(val_accuracy)

    experiment.log_metric("avg_test_loss", avg_test_loss)
    experiment.log_metric("avg_test_accuracy", avg_test_accuracy)

    experiment.log_metric("avg_val_loss", avg_val_loss)
    experiment.log_metric("avg_val_accuracy", avg_val_accuracy)

    # end experiment
    save_model_and_experiment(experiment=experiment, model=model)