import tensorflow as tf
import numpy as np
import json
import ast

# defining parameter ranges for each model (config for optimization of hyperparameters)
def generate_cnn_config(max_num_conv_layers: int, 
                        max_num_dense_layers: int, 
                        random_seed: int):
    """
    Generates config for optimizer for cnn model, using the maximum number of convolutional
    and dense layers. 
    max_num_conv_layers: maximum number of convolutional layers
    max_num_dense_layers: maximum number of dense layers
    random_seed: used for optimizer seed
    """

    cnn_config = {}
    cnn_config["algorithm"] = "bayes"

    parameters = {
        "batch_size": {"type": "discrete", "values": [8, 16, 32, 64, 128]},
        "learning_rate": {"type": "float", "min": 1e-5, "max": 1e-2, "scale": "loguniform"},
        "num_conv_layers": {"type": "integer", "min": 1, "max": max_num_conv_layers},
        "num_dense_layers": {"type": "integer", "min": 1, "max": max_num_dense_layers},
        "dropout": {"type": "float", "min": 0.0, "max": 0.5},
        "pooling":{"type": "categorical", "values": ["max", "avg"]},
        "optimizer": {"type": "categorical", "values": ["adadelta","adam", "adamw", "lion", "rmsprop", "sgd"]},
    }
    for i in range(max_num_conv_layers):
        parameters[f"filters_{i}"] = {"type": "discrete", "values": [16,32,64,128]}
        parameters[f"conv_activation_{i}"] = {"type": "categorical", "values": ["tanh", "sigmoid", "relu"]}
        parameters[f"kernel_size_{i}"] = {"type": "categorical", "values": ["(3,1)","(3,2)","(3,3)","(5,5)"]}

    for i in range(max_num_dense_layers):
        parameters[f"units_{i}"] = {"type": "discrete", "values": [32, 64, 128, 256]}
        parameters[f"dense_activation_{i}"] = {"type": "categorical", "values": ["tanh", "sigmoid", "relu"]}

    cnn_config["parameters"] = parameters
    cnn_config["spec"] = {
        "metric": "avg_val_loss",
        "objective": "minimize",
        "seed": random_seed
    }
    return cnn_config

def build_cnn_from_experiment(experiment, 
                              input_shape, 
                              output_units: int, 
                              output_activation):
    """
    Builds cnn model using the experiment and extracting the data used to build 
    the model.
    experiment: given experiment with hyperparameters
    input_shape: tuple of int
    output_units: number of output units
    ouput_activation: activation in last layer
    """

    # parameters used independently from each layer
    num_conv_layers = experiment.get_parameter("num_conv_layers")
    num_dense_layers = experiment.get_parameter("num_dense_layers")
    dropout = experiment.get_parameter("dropout")
    pooling = experiment.get_parameter("pooling")

    # collecting the parameters used for every convolutional layer
    conv_layers = []
    for i in range(num_conv_layers):
        conv_layers.append(
            (experiment.get_parameter(f"filters_{i}"),
             eval(experiment.get_parameter(f"kernel_size_{i}")),    # kernel_size is a string is converted to tuple of int
             experiment.get_parameter(f"conv_activation_{i}"))
        )

    # collecting the parameters used for every dense layer
    dense_layers = [] 
    for i in range(num_dense_layers):
        dense_layers.append(
            (experiment.get_parameter(f"units_{i}"),
             experiment.get_parameter(f"dense_activation_{i}"))
        )

    return build_cnn_model(input_shape, conv_layers, dense_layers, dropout, output_units, pooling, output_activation)

def build_cnn_from_json(json_path, input_shape, output_units, output_activation='softmax'):
    """
    Reads CNN configuration from a JSON file and builds a CNN model.

    Args:
        json_path (str): Path to the JSON configuration file.
        input_shape (tuple): Input shape of the model (without channel dimension).
        output_units (int): Number of output units.
        output_activation (str): Activation function for the output layer.

    Returns:
        keras.models.Sequential: A compiled CNN model.
    """
    with open(json_path, 'r') as f:
        config = json.load(f)

    # Extract number of layers
    num_conv_layers = int(config['num_conv_layers'])
    num_dense_layers = int(config['num_dense_layers'])

    # Extract conv layers
    conv_layers = []
    for i in range(num_conv_layers):
        filters = int(config[f'filters_{i}'])
        kernel_size = ast.literal_eval(config[f'kernel_size_{i}'])  # safely parse string tuple
        activation = config[f'conv_activation_{i}']
        conv_layers.append((filters, kernel_size, activation))

    # Extract dense layers
    dense_layers = []
    for i in range(num_dense_layers):
        units = int(config[f'units_{i}'])
        activation = config[f'dense_activation_{i}']
        dense_layers.append((units, activation))

    # Extract other required parameters
    dropout = float(config['dropout'])
    pooling = config['pooling']

    # Build and return the model
    return build_cnn_model(input_shape=input_shape,
                           conv_layers=conv_layers,
                           dense_layers=dense_layers,
                           dropout=dropout,
                           output_units=output_units,
                           pooling=pooling,
                           output_activation=output_activation)

def build_cnn_model(input_shape,
                    conv_layers, 
                    dense_layers, 
                    dropout: float,
                    output_units: int, 
                    pooling,
                    output_activation='softmax'):
    """
    Builds a cnn model using the parameters:
    input_shape: tuple of int
    conv_layers: tuple (filters, kernel size, activation)
    dense_layers: tuple (units, activation)
    dropout: float
    output_units: number of output units
    ouput_activation: activation in last layer
    pooling: "avg" or "max"

    returns: model (keras.models.Sequential)
    
    NOTE: pool_size is always (2,1), padding='same'
    """

    # instatiate model
    model = tf.keras.models.Sequential()

    # not necessary but adds to readability
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    # reshape data
    model.add(tf.keras.layers.Reshape(input_shape + (1,)))

    # add the conv and pool layers
    for filters, kernel_size, activation in conv_layers:
        model.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same'))
        
        if pooling == 'max':
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 1)))
        elif pooling == 'avg':
            model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 1)))

    # flatten the data for the dense layers
    model.add(tf.keras.layers.Flatten())

    # add dense layers
    for units, activation in dense_layers:
        model.add(tf.keras.layers.Dense(units, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout))

    # add output layer
    model.add(tf.keras.layers.Dense(output_units, activation=output_activation))
    return model