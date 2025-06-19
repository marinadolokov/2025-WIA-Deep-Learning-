# import libraries
from comet_ml import Experiment
import tensorflow as tf
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TerminateOnNaN

# import own files
from utils import save_model_and_experiment, get_optimizer, TimeLimitCallback
from comet_user_management import set_comet_user_and_project_name, get_api_key, get_project_name
from db_vae_factory import DB_VAE

# constant parameters
RANDOM_SEED = 42                # random seed used in variations
NUM_RANDOM_SPLITS = 5           # number of random splits
EPOCHS = 50                    # epochs
PATIENCE = 2                    # patience for early stopping
INPUT_SHAPE = (128, 6)          # input shape of training data
OUTPUT_UNITS = 12               # number of labels 
OUTPUT_ACTIVATION='softmax'     # activation used in output layer

set_comet_user_and_project_name("Martin", "06-16-db_vae_vae_weight")


# Load train data
df = pd.read_pickle('train.pickle')
all_train_data = np.stack(df.sensor_data.to_numpy(), dtype=np.float32)
all_train_labels = df.label.to_numpy().astype(np.float32)

# Load test data
df_test = pd.read_pickle('test.pickle')
test_data = np.stack(df_test.sensor_data.to_numpy(), dtype=np.float32)
test_labels = df_test.label.to_numpy().astype(np.float32)

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
    # Compile
    model.compile(
        optimizer=optimizer,
        loss=None,
        metrics=["accuracy", "classification_loss", "gini", "total_loss", "vae_loss"],
    )

    # early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=PATIENCE,
                                                      restore_best_weights=True)
    time_limit = TimeLimitCallback(
        max_time=600,  # 10 minutes total
        max_time_per_epoch={0: 100, 1: 50, 8: 10},
        comet_experiment=experiment
    )

    # Create data generator with debiasing random batch sampling for training
    train_data_generator = model.debiased_random_batch_generator(train_data[num_data_set], train_labels[num_data_set],
                                                                 batch_size)

    # Train the model
    model.fit(train_data_generator, epochs=epochs,
              steps_per_epoch=train_data[num_data_set].shape[0] // params["batch_size"],
              validation_data=(valid_data[num_data_set], valid_labels[num_data_set]),
              callbacks = [early_stopping, time_limit, TerminateOnNaN()])

    # evaluate the model on validation data
    val_accuracy, _, _, val_loss, _ = model.evaluate(valid_data[num_data_set], valid_labels[num_data_set])

    # evaluate the model on test data
    test_accuracy, _, _, test_loss, _ = model.evaluate(test_data, test_labels)

    return test_loss, test_accuracy, val_loss, val_accuracy

# Run with different settings (alpha=0 means sampling from uniform distribution, i.e. no debiasing at all, while
# alpha=1 is the most "aggressive" debiasing in sampling)
for vae_weight in [ 0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    experiment = Experiment(api_key=get_api_key(), project_name=get_project_name())
    experiment.log_metric("vae_weight", vae_weight)
    test_loss     = np.zeros(NUM_RANDOM_SPLITS)
    test_accuracy = np.zeros(NUM_RANDOM_SPLITS)
    val_loss = np.zeros(NUM_RANDOM_SPLITS)
    val_accuracy = np.zeros(NUM_RANDOM_SPLITS)

    # Training hyperparameters
    params = dict(
        batch_size=64,  # Number of random training examples fed in at one time.
        learning_rate=3e-1,  # Learning rate for the optimizer.
        latent_dim=64,  # Number of dimensions in the latent space.
        epochs=50,
        optimizer='adadelta',
    )

    # train model on every split of data
    for i in range(NUM_RANDOM_SPLITS):
        model = DB_VAE(latent_dim=params["latent_dim"], input_shape=(128, 6),
                   conv_layers=[(32, (3, 3), tf.nn.sigmoid), (64, (3, 2), tf.nn.tanh), (64, (5, 3), tf.nn.tanh)],
                   dense_layers = [(256, tf.nn.sigmoid), (128, tf.nn.tanh), (128, tf.nn.tanh)],
                   pooling = 'max', dropout=0.18555482216988328, vae_weight=vae_weight, kl_weight=0.1, alpha=1.0)
        test_loss[i], test_accuracy[i], val_loss[i], val_accuracy[i] = train(model, experiment,
                                                                             EPOCHS, params["batch_size"],
                                                                             params["optimizer"],
                                                                             params["learning_rate"], i)
        print(f"Split {i}:", test_loss[i], test_accuracy[i], val_loss[i], val_accuracy[i])
        experiment.log_metric("test_loss", test_loss[i])
        experiment.log_metric("test_accuracy", test_accuracy[i])
        pred_labels = np.argmax(model.predict(test_data), axis=1)
        experiment.log_confusion_matrix(test_labels.astype(np.uint8), pred_labels.astype(np.uint8))

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