from imports import *

class Generator:
    def __init__(self) -> None:
        pass

    def generator_model(self, input_dim : int, output_dim : int, feature_size : int) -> tf.keras.Model:
        model = tf.keras.Sequential([
            LSTM(units=1024, return_sequences=True, 
                input_shape=(input_dim, feature_size), recurrent_dropout=0.3),
            LSTM(units=512, return_sequences=True, recurrent_dropout=0.3),
            LSTM(units=256, return_sequences=True, recurrent_dropout=0.3),
            LSTM(units=128, return_sequences=True, recurrent_dropout=0.3),
            LSTM(units=64, recurrent_dropout=0.3),
            Dense(32),
            Dense(16),
            Dense(8),
            Dense(units=output_dim)
        ])
        return model
    
    def loss(self, fake_output):
        loss_f = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss = loss_f(tf.ones_like(fake_output), fake_output)
        return loss

class Discriminator:
    def __init__(self) -> None:
        pass

    def discriminator_model(self, input_dim : int) -> tf.keras.Model:
        cnn_net = tf.keras.Sequential()
        cnn_net.add(Conv1D(8, input_shape=(input_dim+1, 1), kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
        cnn_net.add(Conv1D(16, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
        cnn_net.add(Conv1D(32, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
        cnn_net.add(Conv1D(64, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
        cnn_net.add(Conv1D(128, kernel_size=1, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
        #cnn_net.add(Flatten())
        cnn_net.add(LeakyReLU())
        cnn_net.add(Dense(220, use_bias=False))
        cnn_net.add(LeakyReLU())
        cnn_net.add(Dense(220, use_bias=False, activation='relu'))
        cnn_net.add(Dense(1, activation='sigmoid'))
        return cnn_net

    def loss(self, real_output, fake_output):
        loss_f = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = loss_f(tf.ones_like(real_output), real_output)
        fake_loss = loss_f(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

class Model:
    def __init__(self, learning_rate, epochs, x_train, output_dim):
        self.lr = learning_rate
        self.epochs = epochs
        self.x_train = x_train
        self.output_dim = output_dim

    def make_model(self):
        print("Generator and Discriminator model")
        g_optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr)
        d_optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr)

        gen = Generator()
        disc = Discriminator()

        generator = gen.generator_model(self.x_train.shape[1], self.output_dim, self.x_train.shape[2])
        discriminator = disc.discriminator_model(self.x_train.shape[1])
        return generator, discriminator, g_optimizer, d_optimizer