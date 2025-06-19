import warnings

from comet_ml import Experiment
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from copy import deepcopy

# Local imports
from comet_user_management import set_comet_user_and_project_name, get_project_name, get_api_key
from db_vae_factory import DB_VAE
from fcn_factory import build_fcn_from_json
from lstm_factory import build_lstm_from_json
from cnn_factory import build_cnn_from_json
from utils import save_model_and_experiment, get_optimizer, TimeLimitCallback
from Preprocess import PREPROCESS_METHODS
from data_augmentation import AUGMENTATION_METHODS

# Global variables
NUM_RANDOM_SPLITS = 5           # Number of random splits
RANDOM_SEED = 42                # Random seed used for random split
PATIENCE = 5                    # Patience used for training
INPUT_SHAPE = (128, 6)          # Input shape of training data
OUTPUT_UNITS = 12               # Number of labels
OUTPUT_ACTIVATION='softmax'     # Activation used in output layer
EPOCHS = 100                      # Epochs

def train_all_splits(model_type, model_json, augmentation_method=lambda x:x, preprocessing_method=lambda x:x):
    # Load training data
    df_train = pd.read_pickle('train.pickle')

    # Augment or preprocess
    df_train = preprocessing_method(augmentation_method(df_train))

    # Extract data and labels
    all_train_data = np.stack(df_train.sensor_data.to_numpy(), dtype=np.float32)
    all_train_labels = df_train.label.to_numpy().astype(np.float32)

    # Random split
    train_data = []
    train_labels = []
    valid_data = []
    valid_labels = []

    # Load test data
    df_test = pd.read_pickle('test.pickle')
    # Augment or preprocess
    df_test = preprocessing_method(df_test)
    test_data = np.stack(df_test.sensor_data.to_numpy(), dtype=np.float32)
    test_labels = df_test.label.to_numpy().astype(np.float32)
    for i in range(NUM_RANDOM_SPLITS):
        new_train_data, new_valid_data, new_train_labels, new_valid_labels =train_test_split(all_train_data, all_train_labels, test_size=0.2, random_state=RANDOM_SEED+i, shuffle=True)

        train_data.append(new_train_data)
        valid_data.append(new_valid_data)
        train_labels.append(new_train_labels)
        valid_labels.append(new_valid_labels)

    experiment = Experiment(
        api_key=get_api_key(),
        project_name=get_project_name(),
    )

    preprocessing_tag = preprocessing_method.__name__
    if preprocessing_tag == "<lambda>":
        preprocessing_tag = "No preprocessing"

    augmentation_tag = augmentation_method.__name__
    if augmentation_tag == "<lambda>":
        augmentation_tag = "No augmentation"

    experiment.add_tag(preprocessing_tag)
    experiment.add_tag(augmentation_tag)
    test_loss = np.zeros(NUM_RANDOM_SPLITS)
    test_accuracy = np.zeros(NUM_RANDOM_SPLITS)
    val_loss = np.zeros(NUM_RANDOM_SPLITS)
    val_accuracy = np.zeros(NUM_RANDOM_SPLITS)

    if model_type != "DBVAE":
        with open(model_json, 'r') as f:
            data = json.load(f)
        # Extract values and convert to appropriate types
        batch_size = int(data['batch_size'])
        learning_rate = float(data['learning_rate'])
        optimizer = data['optimizer'].lower()

    # Train model on every split of data
    for i in range(NUM_RANDOM_SPLITS):
        if model_type == "FCN":
            model = build_fcn_from_json(model_json,
                                input_shape=INPUT_SHAPE,
                                output_units=OUTPUT_UNITS,
                                output_activation=OUTPUT_ACTIVATION)
        elif model_type == "CNN":
            model = build_cnn_from_json(model_json,
                                        input_shape=INPUT_SHAPE,
                                        output_units=OUTPUT_UNITS,
                                        output_activation=OUTPUT_ACTIVATION)
        elif model_type == "LSTM":
            model = build_lstm_from_json(model_json,
                                         input_shape=INPUT_SHAPE,
                                         output_units=OUTPUT_UNITS,
                                         output_activation=OUTPUT_ACTIVATION)
        elif model_type == "DBVAE":
            warnings.warn("Loading a DBVAE model from json is not implemented, so we hardcoded batch size, learning rate and optimizer.")
            # Training hyperparameters
            params = dict(
                batch_size=64,  # Number of random training examples fed in at one time.
                learning_rate=3e-1,  # Learning rate for the optimizer.
                latent_dim=64,  # Number of dimensions in the latent space.
                epochs=50,
                optimizer='adadelta',
            )
            model = DB_VAE(latent_dim=params["latent_dim"], input_shape=(128, 6),
                           conv_layers=[(32, (3, 3), tf.nn.sigmoid), (64, (3, 2), tf.nn.tanh),
                                        (64, (5, 3), tf.nn.tanh)],
                           dense_layers=[(256, tf.nn.sigmoid), (128, tf.nn.tanh), (128, tf.nn.tanh)],
                           pooling='max', dropout=0.18555482216988328, vae_weight=1, kl_weight=0.1, alpha=1.0)
            test_loss[i], test_accuracy[i], val_loss[i], val_accuracy[i] = train(experiment, model,
                                                                                 train_data[i], train_labels[i],
                                                                                 valid_data[i], valid_labels[i],
                                                                                 test_data, test_labels,
                                                                                 EPOCHS, params["batch_size"],
                                                                                 params["optimizer"],
                                                                                 params["learning_rate"])
        if model_type != "DBVAE":
            test_loss[i], test_accuracy[i], val_loss[i], val_accuracy[i] = train(experiment, model,
                                                                             train_data[i], train_labels[i],
                                                                             valid_data[i], valid_labels[i],
                                                                             test_data, test_labels,
                                                                             EPOCHS, batch_size,
                                                                             optimizer, learning_rate)

        experiment.log_metric("test_loss", test_loss[i])
        experiment.log_metric("test_accuracy", test_accuracy[i])

    # Compute averages and log them to comet
    avg_test_loss = np.average(test_loss)
    avg_test_accuracy  = np.average(test_accuracy)
    avg_val_loss = np.average(val_loss)
    avg_val_accuracy  = np.average(val_accuracy)
    experiment.log_metric("avg_test_loss", avg_test_loss)
    experiment.log_metric("avg_test_accuracy", avg_test_accuracy)
    experiment.log_metric("avg_val_loss", avg_val_loss)
    experiment.log_metric("avg_val_accuracy", avg_val_accuracy)

    # End experiment and save it under models
    save_model_and_experiment(experiment=experiment,model=model)

def train(experiment,
          model,
          train_data,
          train_labels,
          valid_data,
          valid_labels,
          test_data,
          test_labels,
          epochs: int,
          batch_size: int,
          optimizer: str,
          learning_rate: float):
    """
    Trains a given model.
    model: given model
    epochs: number of epochs
    batch_size: size of batches
    optimizer: optimizer used in model compilation
    learnig_rate: learning rate for optimizer
    num_data_set: number that specifies which random split is trained
    """
    # Compile model
    optimizer = get_optimizer(optimizer, learning_rate)
    loss=tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=PATIENCE, restore_best_weights=True)
    time_limit = TimeLimitCallback(
        max_time=300,  # 5 minutes total
        comet_experiment=experiment
    )

    # Train the model
    model.fit(train_data, train_labels, batch_size, epochs, validation_data=(valid_data, valid_labels), callbacks = [early_stopping, time_limit])

    # Evaluate the model on validation data
    if(model_type == "DBVAE"):
        val_accuracy, _, _, val_loss, _ = model.evaluate(valid_data, valid_labels)
    else:
        val_loss, val_accuracy = model.evaluate(valid_data, valid_labels)

    # Evaluate the model on test data
    if(model_type == "DBVAE"):
        test_accuracy, _, _, test_loss, _ = model.evaluate(test_data, test_labels)
    else:
        test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    return (test_loss, test_accuracy, val_loss, val_accuracy)

if __name__ == '__main__':
    set_comet_user_and_project_name("Marina", "Test_BEAM")

    model_types = ["FCN", "CNN", "LSTM", "DBVAE"]
    print(f"Type the model type you want to use (must be one of the following names {set(model_types)}):")
    valid_input = False
    model_type = None
    while not valid_input:
        model_type = input().upper()
        if model_type in ["Q", "QUIT"]:
            raise Exception('User quit')
        if model_type in model_types:
            valid_input = True
        else:
            print(f"Invalid answer. Type the model type you want to use (must be one of the following names {set(model_types)}) or 'q' to quit:")

    # Load model json
    subdir = "best_models_json_files"
    full_path = os.path.join(os.getcwd(), subdir)
    json_name_list = list(filter(lambda f: f.startswith("json_" + model_type), os.listdir(full_path)))
    if json_name_list:
        json_name = json_name_list[0]
        json_path = os.path.join(full_path, json_name)
    else:
        json_path = None


    print("Do you want to train the model (1) or just evaluate it (2)? Type '1' or '2':")
    valid_input = False
    while not valid_input:
        answer = input().upper()
        if answer in ["Q", "QUIT"]:
            raise Exception('User quit')
        if answer in ["1", "2"]:
            valid_input = True
        else:
            print("Invalid answer. Do you want to train the model (1) or just evaluate it (2)? Type '1' or '2':")

    if answer == "1":
        print("New models with a similar architecture to the chosen model will be trained (random split validation).")
        print("Do you want to train the model with preprocessing (1), data augmentation (2) or without (3) or both (4)? Type '1', '2', '3' or '4':")
        valid_input = False
        while not valid_input:
            answer = input().upper()
            if answer in ["Q", "QUIT"]:
                raise Exception('User quit')
            if answer in ["1", "2", "3", "4"]:
                valid_input = True
            else:
                print("Invalid answer. Do you want to train the model with preprocessing (1), data augmentation (2) or without (3)? Type '1', '2' or '3':")

        if answer == "1":
            print("We will try and compare all data preprocessing methods...")
            for preprocessing_method in PREPROCESS_METHODS:
                train_all_splits(model_type=model_type, preprocessing_method=preprocessing_method,
                                 model_json=json_path)
        elif answer == "2":
            print("We will try and compare all data augmentation methods...")
            for augmentation_method in AUGMENTATION_METHODS:
                train_all_splits(model_type=model_type, augmentation_method=augmentation_method, model_json=json_path)
        elif answer == '3':
            print("We will try and train the model...")
            train_all_splits(model_type=model_type, model_json=json_path)
        elif answer == '4':
            print("We will try and compare all data preprocessing methods and data augmentation methods...")
            for preprocessing_method in PREPROCESS_METHODS:
                for augmentation_method in AUGMENTATION_METHODS:
                    train_all_splits(model_type=model_type, preprocessing_method=preprocessing_method,
                                     model_json=json_path, augmentation_method=augmentation_method)

    elif answer == '2':
        experiment = Experiment(
            api_key=get_api_key(),
            project_name=get_project_name(),
        )

        # Load model
        subdir = "best_models"
        full_path = os.path.join(os.getcwd(), subdir)
        model_file = list(filter(lambda f: f.startswith("Best_" + model_type), os.listdir(full_path)))[0]
        model = tf.keras.models.load_model(os.path.join(full_path, model_file))

        df_test = pd.read_pickle('test.pickle')
        test_data = np.stack(df_test.sensor_data.to_numpy())
        test_labels = df_test.label.to_numpy().astype('int64')

        experiment.set_name("evaluation of model")
        test_loss, test_accuracy = model.evaluate(test_data, test_labels)
        experiment.log_metric("test_loss", test_loss)
        experiment.log_metric("test_accuracy", test_accuracy)

#zu comet loggen
#beste Modelle laden und evaluieren (auf test daten)
#beste Modell noch einmal trainieren (mit clone, mit random split)
#beste modelle mit preprocessing und data augmentation trainieren (alle funktionen immer in schleife)
