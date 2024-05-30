import numpy as np
import csv
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Loading data from CSV file
def readData():
    inputs = []  # Inputs (joint angles q1 to q7)
    outputs = []  # Outputs (positions x, y, z and orientations alpha, beta, gamma)

    with open('panda_data2_30.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            inputs.append([float(angle) for angle in row[:7]])  # Add inputs (joint angles)
            outputs.append([float(coord) for coord in row[7:]])  # Add outputs (positions and orientations)

    return np.array(inputs), np.array(outputs)

# Loading and preparing data
inputs, outputs = readData()

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

# Data normalization
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Define neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(7,)),   # 7 inputs (q1 to q7)
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(6)  # 6 outputs (positions and orientations)
])

# Compiling the model with accuracy and validation accuracy metrics
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Training the model
history = model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=32, validation_split=0.2)

# Evaluating the model on test data
loss, acc = model.evaluate(X_test_scaled, y_test_scaled)
print("Test loss:", loss)
print("Test accuracy:", acc)

# Plotting accuracy graph of the model
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label ='Testing')
plt.title('Model Accuracy Graph during Training and Testing')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(['Training','Testing'], fontsize=7, loc='lower right')
plt.show()

# Plotting loss graph of the model
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label ='Testing')
plt.title('Model Loss Graph during Training and Testing')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend(['Training','Testing'], fontsize=7, loc='upper right')
plt.show()
