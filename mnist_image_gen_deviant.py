import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, Input, Reshape, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load the trained generator
from tensorflow.keras.models import load_model

generator = load_model('mnist_generator.h5')

def generate_digit(digit):
    target_digit = np.zeros(10)
    target_digit[digit] = 1
    target_digit = np.expand_dims(target_digit, axis=0)  # Add a new axis to match dimensions
    noise = np.random.normal(0, 1, (1, 90))
    input_data = np.hstack([target_digit, noise])
    generated_image = generator.predict(input_data)
    plt.imshow(generated_image[0], cmap='gray')
    plt.show()

while True:
    user_input = input("Enter a digit (0-9) or 'q' to quit: ")
    
    if user_input.lower() == 'q':
        break

    try:
        user_digit = int(user_input)
        if 0 <= user_digit <= 9:
            generate_digit(user_digit)
        else:
            print("Invalid input. Please enter a digit between 0 and 9 or 'q' to quit.")
    except ValueError:
        print("Invalid input. Please enter a digit between 0 and 9 or 'q' to quit.")
