import tensorflow as tf
from comet_ml import Experiment

def generate_lstm_config(max_num_lstm_layers: int,
                         max_num_lstm_units: int,
                         max_num_dense_layers: int,
                         random_seed: int,
                         metric_to_be_minimized: str,
                         resume_with_optimizer_id = None):
    """
    Generates config for optimizer for lstm model, using the maximum number of lstm
    and dense layers and the maximum number of lstm units. 
    max_num_lstm_layers: maximum number of lstm layers
    max_num_lstm_units: maximum number of lstm units
    max_num_dense_layers: maximum number of dense layers
    random_seed: used for optimizer seed
    """
    lstm_config = {}
    lstm_config["algorithm"] = "bayes"

    parameters = {
        "batch_size": {"type": "discrete", "values": [8, 16, 32, 64, 128, 256]},
        "learning_rate": {"type": "float", "min": 1e-5, "max": 1e-2, "scale": "loguniform"},
        "num_lstm_layers": {"type": "integer", "min": 1, "max": max_num_lstm_layers},
        "num_dense_layers": {"type": "integer", "min": 1, "max": max_num_dense_layers},
        "dropout": {"type": "float", "min": 0.0, "max": 0.5},
        "recurrent_dropout": {"type": "float", "min": 0.0, "max": 0.5},
        "optimizer": {"type": "categorical", "values": ["adadelta", "adam", "adamw", "lion", "rmsprop", "sgd"]}
    }
    for i in range(max_num_lstm_layers):
        parameters[f"units_{i}"] = {"type": "integer", "min": 1, "max": max_num_lstm_units}
        parameters[f"activation_{i}"] = {"type": "categorical", "values": ["tanh", "sigmoid"]}
        parameters[f"recurrent_activation_{i}"] = {"type": "categorical", "values": ["tanh", "sigmoid"]}
        parameters[f"go_backwards_{i}"] = {"type": "categorical", "values": ["True", "False"]}

    for i in range(max_num_dense_layers):
        parameters[f"units_dense_{i}"] = {"type": "discrete", "values": [32, 64, 128, 256]}
        parameters[f"activation_dense_{i}"] = {"type": "categorical", "values": ["tanh", "sigmoid", "relu"]}

    lstm_config["parameters"] = parameters
    lstm_config["spec"] = {
        "metric": metric_to_be_minimized,
        "objective": "minimize",
        "seed": random_seed
    }
    if resume_with_optimizer_id is not None:
        lstm_config["spec"]["optimizer_id"] = resume_with_optimizer_id
        lstm_config["spec"]["resume"] = True
    return lstm_config

def build_lstm_model_from_experiment(experiment, 
                                     input_shape, 
                                     output_units: int, 
                                     output_activation='softmax'):
    """
    Builds fcn model using the experiment and extracting the data used to build 
    the model.
    experiment: given experiment with hyperparameters
    input_shape: tuple of int
    output_units: number of output units
    ouput_activation: activation in last layer
    """
    # parameters used independently from each layer
    num_lstm_layers = experiment.get_parameter("num_lstm_layers")
    num_dense_layers = experiment.get_parameter("num_dense_layers")
    dropout = experiment.get_parameter("dropout")
    recurrent_dropout = experiment.get_parameter("recurrent_dropout")

    # collecting the parameters used for every convolutional layer
    lstm_layers = []
    for i in range(num_lstm_layers):
        lstm_layers.append(
            (experiment.get_parameter(f"units_{i}"),
             experiment.get_parameter(f"activation_{i}"),
             experiment.get_parameter(f"recurrent_activation_{i}"),
             dropout, recurrent_dropout,
             bool(experiment.get_parameter(f"go_backwards_{i}"))
             )
        )

    # collecting the parameters used for every dense layer
    dense_layers = []
    for i in range(num_dense_layers):
        dense_layers.append(
            (experiment.get_parameter(f"units_dense_{i}"),
             experiment.get_parameter(f"activation_dense_{i}"))
        )

    return build_lstm_model(input_shape, lstm_layers, dense_layers, output_units, output_activation)

import json
import tensorflow as tf

def build_lstm_from_json(json_path, input_shape, output_units=12, output_activation='softmax'):
    """
    Builds an LSTM model using hyperparameters from a JSON file.

    Args:
        json_path (str): Path to the JSON file.
        input_shape (tuple): Input shape for the model.
        output_units (int): Number of units in the output layer.
        output_activation (str): Activation function for the output layer.

    Returns:
        keras.models.sequential: Compiled LSTM model.
    """
    with open(json_path, 'r') as f:
        config = json.load(f)

    # Extract number of layers
    num_lstm_layers = int(config['num_lstm_layers'])
    num_dense_layers = int(config['num_dense_layers'])

    # Extract LSTM layers
    lstm_layers = []
    for i in range(num_lstm_layers):
        units = int(config[f'units_{i}'])
        activation = config[f'activation_{i}']
        recurrent_activation = config[f'recurrent_activation_{i}']
        dropout = float(config['dropout'])
        recurrent_dropout = float(config['recurrent_dropout'])
        go_backwards = config[f'go_backwards_{i}']
        lstm_layers.append((units, activation, recurrent_activation, dropout, recurrent_dropout, go_backwards))

    # Extract Dense layers
    dense_layers = []
    for i in range(num_dense_layers):
        units = int(config[f'units_dense_{i}'])
        activation_dense = config[f'activation_dense_{i}']
        dense_layers.append((units, activation_dense))

    # Build model
    model = build_lstm_model(input_shape=input_shape,
                             lstm_layers=lstm_layers,
                             dense_layers=dense_layers,
                             output_units=output_units,
                             output_activation=output_activation)

    return model

def build_lstm_model(input_shape,
                     lstm_layers,
                     dense_layers,
                     output_units=12,
                     output_activation='softmax'):
    """
    Builds a lstm model using the parameters:
    input_shape: tuple of int
    lstm_layers: tuple (units, activation, recurrent_activation, dropout, recurrent_dropout, go_backwards)
    dense_layers: tuple (units, activation)
    output_units: number of output units
    ouput_activation: activation in last layer

    returns: model (keras.models.Sequential)
    """

    # instantiate model
    model = tf.keras.models.Sequential()

    # not necessary but adds to readability
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    # add all lstm layers but the last one (return the whole sequence for the next lstm)
    # TODO: add flag for forward or backward pass of the lstm layer (maybe alternating layers help?)
    for units, activation, recurrent_activation, dropout, recurrent_dropout, go_backwards  in lstm_layers[:-1]:
        model.add(tf.keras.layers.LSTM(units=units, activation=activation, recurrent_activation=recurrent_activation,
                                       dropout=dropout, recurrent_dropout=recurrent_dropout, go_backwards=go_backwards,
                                       return_sequences=True))

    # add last lstm layer (returns only the last output)
    units, activation, recurrent_activation, dropout, recurrent_dropout, go_backwards = lstm_layers[-1]
    model.add(tf.keras.layers.LSTM(units=units, activation=activation, recurrent_activation=recurrent_activation,
                                   dropout=dropout, recurrent_dropout=recurrent_dropout, go_backwards=go_backwards,
                                   return_sequences=False))

    # add dense layers
    for units, activation in dense_layers:
        model.add(tf.keras.layers.Dense(units, activation=activation))

    # add output layer
    model.add(tf.keras.layers.Dense(output_units, activation=output_activation))

    return model