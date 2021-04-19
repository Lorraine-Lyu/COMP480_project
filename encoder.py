import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, Input
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

latent_dim = 32 

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    # input = Input(shape=(32,))
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Dense(latent_dim, activation='relu', input_shape=(32,)),
    ])
    # self.decoder = tf.keras.Sequential([
    #   layers.Dense(784, activation='sigmoid'),
    #   layers.Reshape((64))
    # ])

  def call(self, x):
    encoded = self.encoder(x)
    print(encoded)
    # decoded = self.decoder(encoded)
    return encoded

autoencoder = Autoencoder(latent_dim)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

