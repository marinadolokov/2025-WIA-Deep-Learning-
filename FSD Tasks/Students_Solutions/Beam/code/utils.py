import tensorflow as tf
import time
import warnings
import numpy as np

def generate_model_name(model, experiment_id):
    """
    Generates a model name like '2xDense_Lstm_Dense-id_<experiment_id>',
    by grouping consecutive layers of the same type.
    Only Conv2D, LSTM, and Dense layers are considered.
    """
    # Mapping from layer types to short names
    layer_map = {
        'Conv2D': 'Conv',
        'Dense': 'Dense',
        'LSTM': 'Lstm'
    }
    relevant_types = set(layer_map.keys())

    # Extract the layer types in order, filtering only relevant ones
    layer_types = [type(layer).__name__ for layer in model.layers if type(layer).__name__ in relevant_types]

    if not layer_types:
        return f"models/Model(Unknown)-id_{experiment_id}"

    # Group consecutive layers of the same type
    name_parts = []
    prev_type = None
    count = 0
    for ltype in layer_types + [None]:  # Add dummy None at the end to flush the last group
        if ltype == prev_type:
            count += 1
        else:
            if prev_type is not None:
                prefix = f"{count}x" if count > 1 else ""
                name_parts.append(f"{prefix}{layer_map[prev_type]}")
            prev_type = ltype
            count = 1
    model_name = "models/Model(" + "+".join(name_parts) + f")-id_{experiment_id}"
    return model_name


def save_model_and_experiment(experiment, model):
    """Saves model with standardized naming conventions (including the unique experiment id). Sets the same name for
    the experiment and logs the model to the experiment (as "trained-model")."""
    experiment_id = experiment.get_key()
    experiment_name = generate_model_name(model, experiment_id)
    experiment.set_name(experiment_name)
    model_name = experiment_name + ".keras"
    model.save(model_name)
    experiment.log_model("trained-model", model_name)
    experiment.end()

def get_optimizer(name: str, lr: float):
    optimizers = {
        "adadelta": tf.keras.optimizers.Adadelta,
        "adam": tf.keras.optimizers.Adam,
        "adamw": tf.keras.optimizers.AdamW,
        "lion": tf.keras.optimizers.Lion,
        "rmsprop": tf.keras.optimizers.RMSprop,
        "sgd": tf.keras.optimizers.SGD,
    }

    return optimizers[name](learning_rate=lr)


class TimeLimitCallback(tf.keras.callbacks.Callback):
    def __init__(self, max_time=None, max_time_per_epoch=None, comet_experiment=None):
        """
        Args:
            max_time (float): Maximum total training time in seconds.
            max_time_per_epoch (dict): Mapping {epoch_idx: max_seconds}.
            comet_experiment: Optional Comet experiment object.
        """
        super().__init__()
        self.max_time = max_time
        self.max_time_per_epoch = max_time_per_epoch or {}
        self.comet_experiment = comet_experiment
        self._train_start_time = None
        self._epoch_start_time = None

    def on_train_begin(self, logs=None):
        self._train_start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # Get epoch duration from Comet if possible
        # epoch_duration = None
        # if self.comet_experiment is not None:
        #     try:
        #         # Comet logs metrics as a list of dicts; get last epoch's duration
        #         metrics = self.comet_experiment.get_metrics_summary("epoch_duration")
        #         if isinstance(metrics, dict) and "valueMax" in metrics:
        #             # valueMax is the latest value logged
        #             epoch_duration = float(metrics["valueMax"])
        #     except Exception as e:
        #         warnings.warn(f"Could not retrieve epoch_duration from Comet: {e}")

        # # Fallback to local timing if Comet value not available
        # if epoch_duration is None:
        #     epoch_duration = time.time() - self._epoch_start_time

        # Get epoch duration
        epoch_duration = time.time() - self._epoch_start_time

        # Check per-epoch time limit
        if epoch in self.max_time_per_epoch.keys():
            if epoch_duration > self.max_time_per_epoch[epoch]:
                warnings.warn(
                    f"Epoch {epoch}: exceeded max_time_per_epoch "
                    f"({epoch_duration:.1f}s > {self.max_time_per_epoch[epoch]}s). Stopping training."
                )
                self.model.stop_training = True

        # Check global max_time
        if self.max_time is not None:
            elapsed_total = time.time() - self._train_start_time
            if elapsed_total > self.max_time:
                warnings.warn(
                    f"Training: exceeded max_time "
                    f"({elapsed_total:.1f}s > {self.max_time}s). Stopping training."
                )
                self.model.stop_training = True

    def on_batch_end(self, batch, logs=None):
        # Optionally, you could check time here for even finer control (per batch)
        if self.max_time is not None:
            elapsed_total = time.time() - self._train_start_time
            if elapsed_total > self.max_time:
                warnings.warn(
                    f"Training: exceeded max_time during batch "
                    f"({elapsed_total:.1f}s > {self.max_time}s). Stopping training."
                )
                self.model.stop_training = True


def normalized_gini(array):
    """Calculate the normalized Gini index for a numpy array."""
    array = array.flatten()
    if np.any(array < 0):
        array -= np.min(array)
    array = array + 1e-7  # Avoid zeros
    array = np.sort(array)
    n = len(array)
    index = np.arange(1, n+1)
    gini = np.sum((2 * index - n - 1) * array) / (n * np.sum(array))
    max_gini = (n - 1) / n
    return gini / max_gini if max_gini > 0 else 0.0


if __name__ == "__main__":
    # from tensorflow.keras import Sequential
    # from tensorflow.keras.layers import Dense, LSTM, Conv2D
    # model2 = Sequential([
    #     Conv2D(32, (3, 3), input_shape=(28, 28, 1)),
    #     Conv2D(64, (3, 3)),
    #     Conv2D(128, (3, 3)),
    #     Dense(50)
    # ])
    # print(generate_model_name(model2, "xyz789"))

    print(normalized_gini(np.array([0, 0, 1])))
    print(normalized_gini(np.array([1, 1, 1])))