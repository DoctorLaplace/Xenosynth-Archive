import numpy as np
from tensorflow.keras.models import load_model

# Load the trained models
xor_model = load_model('xor_model.h5')
nand_model = load_model('nand_model.h5')

def neural_xor(x1, x2):
    input_data = np.array([[int(x1), int(x2)]])
    output = xor_model.predict(input_data)
    return round(output[0][0])

def neural_nand(x1, x2):
    input_data = np.array([[int(x1), int(x2)]])
    output = nand_model.predict(input_data)
    return round(output[0][0])

# Test the neural_xor function
print("Neural XOR gate predictions:")
print(f"False XOR False = {neural_xor(False, False)}")
print(f"False XOR True = {neural_xor(False, True)}")
print(f"True XOR False = {neural_xor(True, False)}")
print(f"True XOR True = {neural_xor(True, True)}")

# Test the neural_nand function
print("\nNeural NAND gate predictions:")
print(f"False NAND False = {neural_nand(False, False)}")
print(f"False NAND True = {neural_nand(False, True)}")
print(f"True NAND False = {neural_nand(True, False)}")
print(f"True NAND True = {neural_nand(True, True)}")
