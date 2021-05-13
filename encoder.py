from tensorflow import keras
from tensorflow.keras import layers, regularizers

latent_dim = 32


class Autoencoder(keras.models.Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = keras.Sequential(
            [
                layers.Dense(256, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(self.latent_dim, activation="relu")
                # layers.Dense(self.latent_dim, activation="relu", activity_regularizer=regularizers.l1(10e-3)),
            ]
        )

        self.decoder = keras.Sequential(
            [
                layers.Dense(64, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(512, activation="relu"),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


model = Autoencoder(latent_dim)
model.build((None, 512))
model.compile(
    loss="mean_squared_error",
    optimizer="adam",
    metrics=["accuracy"],
)
