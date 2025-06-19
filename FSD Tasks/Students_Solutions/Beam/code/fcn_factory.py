import tensorflow as tf
import json

def generate_fcn_config(max_num_hidden_layers, random_seed):
    """
    Generates config for optimizer for fcn model, using the maximum number of 
    hidden layers. 
    max_num_hidden_layers: maximum number of hidden layers
    random_seed: used for optimizer seed
    """

    fcn_config = {}
    fcn_config["algorithm"] = "bayes"

    parameters = {
        "batch_size": {"type": "discrete", "values": [8, 16, 32, 64, 128]},
        "learning_rate": {"type": "float", "min": 1e-5, "max": 1e-2, "scale": "loguniform"},
        "num_hidden_layers": {"type": "integer", "min": 1, "max": max_num_hidden_layers},
        "dropout": {"type": "float", "min": 0.0, "max": 0.5},
        "optimizer": {"type": "categorical", "values": ["adadelta","adam", "adamw", "lion", "rmsprop", "sgd"]},
    }

    for i in range(max_num_hidden_layers):
        parameters[f"units_{i}"] = {"type": "discrete", "values": [32, 64, 128, 256]}
        parameters[f"activation_{i}"] = {"type": "categorical", "values": ["tanh", "sigmoid", "relu"]}
        
    fcn_config["parameters"] = parameters
    fcn_config["spec"] = {
        "metric": "avg_val_loss",
        "objective": "minimize",
        "seed": random_seed
    }
    return fcn_config

def build_fcn_from_experiment(experiment, 
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
    num_hidden_layers = experiment.get_parameter("num_hidden_layers")
    dropout = experiment.get_parameter("dropout")

    # collecting the parameters used for every layer
    hidden_layers = []
    for i in range(num_hidden_layers):
        hidden_layers.append(
            (experiment.get_parameter(f"units_{i}"),
             experiment.get_parameter(f"activation_{i}"))
        )

    return build_fcn_model(input_shape, hidden_layers, dropout, output_units, output_activation)


def build_fcn_from_json(json_path, input_shape, output_units, output_activation='softmax'):
    """
    Reads a JSON config and builds a fully connected neural network model.

    Parameters:
        json_path (str): Path to the JSON configuration file.
        input_shape (tuple): Shape of the input (excluding batch dimension).
        output_units (int): Number of output units.
        output_activation (str): Activation function for the output layer.

    Returns:
        model: A compiled tf.keras model.
    """

    # Load the JSON config
    with open(json_path, 'r') as f:
        config = json.load(f)

    # Parse number of hidden layers
    num_hidden_layers = int(config['num_hidden_layers'])

    # Parse hidden layers (units and activations)
    hidden_layers = []
    for i in range(num_hidden_layers):
        units_key = f'units_{i}'
        activation_key = f'activation_{i}'
        if units_key in config and activation_key in config:
            units = int(config[units_key])
            activation = config[activation_key]
            hidden_layers.append((units, activation))

    # Parse dropout
    dropout = float(config['dropout'])

    # Build the model
    model = build_fcn_model(
        input_shape=input_shape,
        hidden_layers=hidden_layers,
        dropout=dropout,
        output_units=output_units,
        ouput_activation=output_activation
    )

    return model

def build_fcn_model(input_shape,
                    hidden_layers,
                    dropout: float,
                    output_units: int,
                    ouput_activation= 'softmax'):
    """
    Builds a fcn model using the parameters:
    input_shape: tuple of int
    hidden_layers: tuple (units, activation)
    dropout: float
    output_units: number of output units
    ouput_activation: activation in last layer

    returns: model (keras.models.Sequential)
    """

    # instatiate model
    model = tf.keras.models.Sequential()

    # not necessary but adds to readability
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    # flatten the data for the dense layers
    model.add(tf.keras.layers.Flatten())

    # add the hidden layers
    for units, activation in hidden_layers:
        model.add(tf.keras.layers.Dense(units=units, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout))

    # add output layer
    model.add(tf.keras.layers.Dense(units=output_units,activation=ouput_activation))
    return model