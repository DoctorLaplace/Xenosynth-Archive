import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create a simple TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Generate synthetic training data
x_train = np.linspace(-1, 1, 100)
y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.33

# Train the model
history = model.fit(x_train, y_train, epochs=5000, verbose=0)

# Plot the training loss
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Test the model
x_test = np.array([-0.5, 0, 0.5])
y_pred = model.predict(x_test)

print(f'Predictions for {x_test}: {y_pred[:, 0]}')
