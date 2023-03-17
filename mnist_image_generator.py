import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, Input, Reshape, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (_, _) = mnist.load_data()
x_train = x_train / 255.0
x_train = np.expand_dims(x_train, axis=-1)

# Generator model
def create_generator(latent_dim):
    model = Sequential([
        Dense(128, input_dim=latent_dim),
        LeakyReLU(0.2),
        BatchNormalization(momentum=0.8),
        Dense(256),
        LeakyReLU(0.2),
        BatchNormalization(momentum=0.8),
        Dense(512),
        LeakyReLU(0.2),
        BatchNormalization(momentum=0.8),
        Dense(28 * 28, activation='sigmoid'),
        Reshape((28, 28))
    ])
    return model

# Discriminator model
def create_discriminator():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(512),
        LeakyReLU(0.2),
        Dropout(0.5),
        Dense(256),
        LeakyReLU(0.2),
        Dropout(0.5),
        Dense(128),
        LeakyReLU(0.2),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

latent_dim = 100
generator = create_generator(latent_dim)
discriminator = create_discriminator()

# Compile discriminator
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

# GAN model
noise_input = Input(shape=(latent_dim,))
generated_image = generator(noise_input)
discriminator.trainable = False
validity = discriminator(generated_image)
gan = Model(noise_input, validity)
gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# Train GAN
batch_size = 32
epochs = 30000

for epoch in range(epochs):
    # Train discriminator
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_images = x_train[idx]
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    gen_images = generator.predict(noise)
    
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(gen_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, real_labels)
    
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, D loss: {d_loss[0]}, accuracy: {100 * d_loss[1]}, G loss: {g_loss}")

# Save the trained generator
generator.save('mnist_generator.h5')

# Load the trained generator
from tensorflow.keras.models import load_model

generator = load_model('mnist_generator.h5')

def generate_digit(digit):
    target_digit = np.zeros(10)
    target_digit[digit] = 1
    noise = np.random.normal(0, 1, (1, 90))
    input_data = np.hstack([target_digit, noise])
    generated_image = generator.predict(input_data)
    plt.imshow(generated_image[0], cmap='gray')
    plt.show()

# Ask the user to enter a digit
user_digit = int(input("Enter a digit (0-9): "))
generate_digit(user_digit)
