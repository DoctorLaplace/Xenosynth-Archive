import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Define the XOR input and output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# Create a simple neural network model
model = Sequential([
    Dense(2, activation='sigmoid', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the XOR dataset
history = model.fit(X, Y, epochs=5000, verbose=0)

# Evaluate the model on the XOR dataset
_, accuracy = model.evaluate(X, Y, verbose=0)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
model.save('xor_model.h5')


# Load the trained model and use it for predictions
from tensorflow.keras.models import load_model

def xor_gate(x1, x2):
    loaded_model = load_model('xor_model.h5')
    return round(loaded_model.predict(np.array([[x1, x2]]))[0][0])

print("XOR gate predictions:")
print(f"0 XOR 0 = {xor_gate(0, 0)}")
print(f"0 XOR 1 = {xor_gate(0, 1)}")
print(f"1 XOR 0 = {xor_gate(1, 0)}")
print(f"1 XOR 1 = {xor_gate(1, 1)}")