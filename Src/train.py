from imports import *
from model_dispatcher import Generator, Discriminator

class Train:
    def __init__(self) -> None:
        self.stock_name = "TSLA"

    @tf.function
    def train_step(self, real_x, real_y, yc, generator, discriminator, g_optimizer, d_optimizer):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = generator(real_x, training=True)
            generated_data_reshape = tf.reshape(generated_data, [generated_data.shape[0], generated_data.shape[1], 1])
            d_fake_input = tf.concat([tf.cast(generated_data_reshape, tf.float64), yc], axis=1)
            real_y_reshape = tf.reshape(real_y, [real_y.shape[0], real_y.shape[1], 1])
            d_real_input = tf.concat([real_y_reshape, yc], axis=1)

            real_output = discriminator(d_real_input, training=True)
            fake_output = discriminator(d_fake_input, training=True)

            g_loss = Generator.loss(fake_output)
            disc_loss = Discriminator.loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(g_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        g_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        d_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return real_y, generated_data, {'d_loss': disc_loss, 'g_loss': g_loss}
    
    def training(self, real_x, real_y, yc, Epochs, generator, discriminator, g_optimizer, d_optimizer, checkpoint = 50):
        train_info = {}
        train_info["discriminator_loss"] = []
        train_info["generator_loss"] = []

        for epoch in tqdm(range(Epochs)):
            real_price, fake_price, loss = self.train_step(real_x, real_y, yc, generator, discriminator, g_optimizer, d_optimizer)
            G_losses = []
            D_losses = []
            Real_price = []
            Predicted_price = []
            D_losses.append(loss['d_loss'].numpy())
            G_losses.append(loss['g_loss'].numpy())
            Predicted_price.append(fake_price.numpy())
            Real_price.append(real_price.numpy())

            #Save model every X checkpoints
            if (epoch + 1) % checkpoint == 0:
                tf.keras.models.save_model(generator, f'../Models/Generator/{self.stock_name}/generator_V_%d.h5' % epoch)
                tf.keras.models.save_model(discriminator, f'../Models/Discrimintor/{self.stock_name}/discriminator_V_%d.h5' % epoch)
                print('epoch', epoch + 1, 'discriminator_loss', loss['d_loss'].numpy(), 'generator_loss', loss['g_loss'].numpy())
        
            train_info["discriminator_loss"].append(D_losses)
            train_info["generator_loss"].append(G_losses)
    
        Predicted_price = np.array(Predicted_price)
        Predicted_price = Predicted_price.reshape(Predicted_price.shape[1], Predicted_price.shape[2])
        Real_price = np.array(Real_price)
        Real_price = Real_price.reshape(Real_price.shape[1], Real_price.shape[2])

        plots.Utils.training_plot(train_info['discriminator_loss'], train_info['generator_loss'], output_file = 'training_plot.png')

        return Predicted_price, Real_price, np.sqrt(mean_squared_error(Real_price, Predicted_price)) / np.mean(Real_price)