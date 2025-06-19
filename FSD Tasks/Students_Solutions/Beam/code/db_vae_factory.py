import warnings
from math import prod
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd


from cnn_factory import build_cnn_model
from utils import normalized_gini


def vae_loss_function(x, x_recon, mu, logsigma, kl_weight):
  '''Inputs:
  * an input - x,
  * reconstructed output - x_recon,
  * encoded means - mu,
  * encoded log of variance (not standard deviation!) - logsigma,
  * weight parameter for the latent loss - kl_weight

  Outputs:
  * loss value - vae_loss
  '''

  # Define the latent loss.
  latent_loss = 0.5 * tf.reduce_sum(tf.exp(logsigma) + tf.square(mu) - 1.0 - logsigma, axis=1)

  # Define the reconstruction loss as the mean absolute pixel-wise
  # difference between the input and reconstruction.
  reconstruction_loss = tf.reduce_mean(tf.abs(x - x_recon), axis=(1, 2))

  # Define the VAE loss.
  vae_loss = kl_weight * latent_loss + reconstruction_loss

  return vae_loss

def sampling(z_mean, z_logsigma):
  '''Performing the reparameterization trick by sampling from a standard Gaussian distribution.

  Inputs:
  * mean of latent distribution - z_mean,
  * log of variance of latent distribution - z_logsigma

  Outputs:
  * sampled latent vector - z
  '''

  batch = tf.shape(z_mean)[0]  # Gets the actual batch size at runtime
  latent_dim = tf.shape(z_mean)[1]
  epsilon = tf.random.normal(shape=(batch, latent_dim)) # Use these sizes to create a standard Gaussian distribution for sampling.

  # Define the reparameterization computation.
  z = z_mean + tf.math.exp(0.5 * z_logsigma) * epsilon
  return z

def build_decoder(latent_dim, conv_layers_from_encoder):
    """Builds decoder part of the variational autoencoder. It is more or less symmetric to the encoder."""

    # instatiate model
    model = tf.keras.models.Sequential()

    # not necessary but adds to readability
    model.add(tf.keras.layers.InputLayer(shape=(latent_dim,)))

    # calculate the dimension which the encoder input had after all convolution and pooling layers, but before flattening
    num_filters_last_conv = conv_layers_from_encoder[-1][0]
    shape = (128//2**len(conv_layers_from_encoder), 6, num_filters_last_conv)

    # use a Dense and a Reshape layer to transform the vector of latent variables to a tensor of shape
    model.add(tf.keras.layers.Dense(prod(shape), activation='relu'))
    model.add(tf.keras.layers.Reshape(shape))

    # mirror the layers of the decoder
    for filters, kernel_size, activation in reversed(conv_layers_from_encoder):
        model.add(tf.keras.layers.Conv2DTranspose(padding='same', activation=activation, filters=filters,
                                                  kernel_size=kernel_size, strides=(2, 1)))

    # This layer yields a weighted sum of the feature maps from the previous layer -> no mixing of spatial information
    model.add(tf.keras.layers.Conv2DTranspose(padding='same', activation=None, filters=1, kernel_size=1, strides=1))

    model.add(tf.keras.layers.Lambda(lambda a: tf.squeeze(a, axis=-1)))

    return model

# Get the mean of the latent space.
def get_latent_mu(data, dbvae, latent_dim, batch_size):
  '''Inputs:
  * group of data samples - data
  * model to pass data through - dbvae
  * latent space dimensionality - latent_dim
  * number of data samples randomly selected at one time - batch_size

  Outputs:
  * mean of the latent space for our given data - mu
  '''
  N = data.shape[0]
  # N = num_data_samples + (-num_data_samples % batch_size) # Find the number of images provided.
  mu = np.zeros((N, latent_dim)) # Initialize the mean as a multi-dimensional array of zeros.

  for start_ind in range(0, N, batch_size): # Loop through every possible batch in our provided images.
    end_ind = min(start_ind+batch_size, N) # Loop provides starting index, so we need to find the ending index of our batch.
    batch = (data[start_ind:end_ind]).astype(np.float32) # Convert our batch of images to a numerical representation.
    _, batch_mu, _ = dbvae.encode(batch) # Find the latent space mean for our batch by using our DB-VAE encoder.
    mu[start_ind:end_ind] = batch_mu # Add the batch mean to our overall mean variable.

  return mu # Return the entire latent space mean after all batches are done.

def get_training_sample_probabilities(data, dbvae, latent_dim, batch_size, alpha, bins=10):
  '''Function that recomputes the latent space sampling probabilities for the images within a given batch based on their feature distribution in the training data.

  Inputs:
  * group of data samples - data
  * model to pass data through - dbvae
  * latent space dimensionality - latent_dim
  * number of data samples randomly selected at one time - batch_size
  * aggressiveness of debiasing (between 0 and 1) - alpha
  * number of classes for probabilities - bins

  Outputs:
  * new sampling probabilities - training_sample_p
  '''

  warnings.warn("Recomputing the sampling probabilities")

  assert alpha >= 0 and alpha <= 1, "alpha must be between 0 and 1"

  # Get the mean of the latent space for our given data.
  mu = get_latent_mu(data, dbvae, latent_dim, batch_size) # Remember to use the correct inputs for the function.
  num_samples = mu.shape[0]
  training_sample_p = np.zeros(num_samples) # Initialize our new sampling probabilities to be as large as our means.

  for i in range(latent_dim): # Loop through every latent variable in the latent space.
      latent_distribution = mu[:,i] # Get the means for all of the data samples at our given latent variable.


      hist_density, bin_edges = np.histogram(latent_distribution, density=True, bins=bins) # Create a histogram using the latent distribution and our set number of bins.

      # Extend the range of our histogram to be infinite.
      bin_edges[0] = -float('inf')
      bin_edges[-1] = float('inf')

      # Use the np.digitize function to find the bins in the latent distribution that every data sample falls into.
      bin_idx = np.digitize(latent_distribution, bin_edges)

      assert np.all(hist_density[bin_idx-1] > 0), "The probability density for drawing samples is not > 0 everywhere!"

      # Invert the histogram.
      p_inverse = 1.0 / (hist_density[bin_idx-1])

      # Normalize all the inverted probabilities.
      p_inverse = p_inverse / np.sum(p_inverse)

      # Use a convex combination between p_inverse and uniform distribution
      p_uniform = np.full(num_samples, 1 / num_samples)
      p = alpha * p_inverse + (1-alpha) * p_uniform

      # Update the sampling probabilities by comparing the new probability distribution to the existing one.
      # Note: In the original paper they use product instead of maximum, see https://introtodeeplearning.com/AAAI_MitigatingAlgorithmicBias.pdf
      training_sample_p = np.maximum(p, training_sample_p)

  # After the probability distribution is finalized, perform the final smoothing operation on our probability distribution.
  training_sample_p = training_sample_p / np.sum(training_sample_p)

  # Track (normalized) gini inequality index (between 0 and 1) of training_sample_p,
  # i.e. uniform distribution is 0 (minimum) and dirac distribution is 1 (maximum)
  # https://en.wikipedia.org/wiki/Gini_coefficient
  dbvae.sampling_probability_gini_tracker.update_state(normalized_gini(training_sample_p))

  return training_sample_p

@keras.utils.register_keras_serializable()
class DB_VAE(tf.keras.Model):
    def __init__(self, latent_dim, input_shape, conv_layers, dense_layers, dropout, pooling, vae_weight, kl_weight, alpha):
        super(DB_VAE, self).__init__()
        self.latent_dim = latent_dim # Take an input of latent_dim to define the model's latent dimensionality.
        self.input_shape = input_shape
        self.vae_weight = vae_weight
        self.kl_weight = kl_weight
        self.alpha = alpha

        # Define the number of outputs for the encoder. We will also have latent_dim latent variables.
        num_encoder_dims = 2*self.latent_dim + 12

        self.encoder = build_cnn_model(input_shape=input_shape, conv_layers=conv_layers, dense_layers=dense_layers,
                                       dropout=dropout, pooling=pooling, output_units=num_encoder_dims)
        self.decoder = build_decoder(latent_dim=latent_dim, conv_layers_from_encoder=conv_layers)

        # Tracker for metrics of interest
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.classification_loss_tracker = tf.keras.metrics.Mean(name="classification_loss")
        self.vae_loss_tracker = tf.keras.metrics.Mean(name="vae_loss")
        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.sampling_probability_gini_tracker = tf.keras.metrics.Mean(name="gini")

    @property
    def metrics(self):
        return [
            self.accuracy_tracker,
            self.classification_loss_tracker,
            self.sampling_probability_gini_tracker,
            self.total_loss_tracker,
            self.vae_loss_tracker,
        ]

    def encode(self, x):
        # Get the output of the encoder based on its inputs.
        encoder_output = self.encoder(x)
        # Get the actual classifications based on those outputs.
        y_logits = encoder_output[:, :12]

        # Get the mean and log of variance for the latent space.
        z_mean = encoder_output[:, 12:self.latent_dim + 12]
        z_logsigma = encoder_output[:, self.latent_dim + 12:]

        return y_logits, z_mean, z_logsigma

    @staticmethod
    def reparameterized_sampling(z_mean, z_logsigma):
        z = sampling(z_mean, z_logsigma)  # Given the latent space's mean and log of variance, sample some latent variables.
        return z

    def decode(self, z):
        # Use the decoder to take the reparameterized samples and output the reconstruction.
        reconstruction = self.decoder(z)  # Pass the samples through the decoder, get the reconstruction.
        return reconstruction

    # The call function passes an input of x all the way through the DB-VAE.
    def call(self, x):
        # Encode the input to the latent space and get a prediction.
        y_logits, z_mean, z_logsigma = self.encode(x)

        # Perform reparameterization on the latent space.
        z = self.reparameterized_sampling(z_mean, z_logsigma)

        # Find the reconstruction based on the reparameterized samples.
        x_recon = self.decode(z)
        return x_recon, y_logits, z_mean, z_logsigma

    def debiased_random_batch_generator(self, full_data, full_labels, batch_size):
        """Returns a generator that performs debiased random batch sampling every kth training step,
        where k = len(full_data)//batch_size (which often coincides with the number of steps per epoch)."""
        num_samples = len(full_data)
        steps_per_p = num_samples // batch_size
        current_step = 0
        p = np.full(num_samples, 1/num_samples)  # Initialize with uniform distribution
        while True:
            current_step += 1
            idx = np.random.choice(num_samples, size=batch_size, replace=True, p=p)
            yield full_data[idx], full_labels[idx]
            if current_step%steps_per_p == 0:
                p = get_training_sample_probabilities(data=full_data, dbvae=self, latent_dim=self.latent_dim,
                                                      batch_size=batch_size, alpha=self.alpha, bins=10)

    def train_step(self, data):
        """This function allows the use of model.fit(...)."""
        x, y = data

        # Calculate total loss and track the gradient in the process
        with tf.GradientTape() as tape:
            x_recon, y_logits, z_mean, z_logsigma = self(x, training=True)
            vae_loss = vae_loss_function(x, x_recon, z_mean, z_logsigma, kl_weight=self.kl_weight)
            classification_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y, y_logits)
            total_loss = tf.reduce_mean(classification_loss + self.vae_weight * vae_loss)

        # Update weights
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.classification_loss_tracker.update_state(classification_loss)
        self.vae_loss_tracker.update_state(vae_loss)
        self.accuracy_tracker.update_state(y, y_logits)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """This function allows the use of model.evaluate(...)."""
        x, y = data

        # Forward pass (no gradient tape, no weight updates)
        x_recon, y_logits, z_mean, z_logsigma = self(x, training=False)
        vae_loss = vae_loss_function(x, x_recon, z_mean, z_logsigma, kl_weight=self.kl_weight)
        classification_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y, y_logits)
        total_loss = classification_loss + self.vae_weight * vae_loss

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.classification_loss_tracker.update_state(classification_loss)
        self.vae_loss_tracker.update_state(vae_loss)
        self.accuracy_tracker.update_state(y, y_logits)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    # Based on an input of x, predict the label
    def predict(self, x):
        y_logits, z_mean, z_logsigma = self.encode(x) # Use the encoder just like a CNN.
        return y_logits # Return only the prediction without the latent space.


if __name__ == "__main__":
    # Training hyperparameters
    params = dict(
      batch_size = 64, # Number of random training examples fed in at one time.
      learning_rate = 3e-1, # Learning rate for the optimizer.
      latent_dim = 64,  #  Number of dimensions in the latent space.
      epochs = 2
    )

    # Set optimizer
    optimizer = tf.keras.optimizers.Adadelta(params["learning_rate"])

    # Create a new DB-VAE model
    dbvae = DB_VAE(latent_dim=params["latent_dim"], input_shape=(128, 6),
                   conv_layers=[(32, (3, 3), tf.nn.sigmoid), (64, (3, 2), tf.nn.tanh), (64, (5, 3), tf.nn.tanh)],
                   dense_layers = [(256, tf.nn.sigmoid), (128, tf.nn.tanh), (128, tf.nn.tanh)],
                   pooling = 'max', dropout=0.18555482216988328, vae_weight=1, kl_weight=1, alpha=1)

    # Compile
    dbvae.compile(
        optimizer=optimizer,
        loss=None,
        metrics=["accuracy", "classification_loss", "gini", "total_loss", "vae_loss"],
    )

    # Load train data
    df = pd.read_pickle('train.pickle')
    all_train_data = np.stack(df.sensor_data.to_numpy(), dtype=np.float32)
    all_train_labels = df.label.to_numpy().astype(np.float32)

    # Create data generator with debiasing random batch sampling for training
    train_data_generator = dbvae.debiased_random_batch_generator(all_train_data, all_train_labels, params["batch_size"])

    # Train
    dbvae.fit(train_data_generator, epochs=params["epochs"],
              steps_per_epoch= all_train_data.shape[0] // params["batch_size"],)

    # Load test data
    df_test = pd.read_pickle('test.pickle')
    test_data = np.stack(df_test.sensor_data.to_numpy(), dtype=np.float32)
    test_labels = df_test.label.to_numpy().astype(np.float32)

    #Test
    print("Test:")
    dbvae.reset_metrics()
    result = dbvae.evaluate(x=test_data, y=test_labels)
    print(result)