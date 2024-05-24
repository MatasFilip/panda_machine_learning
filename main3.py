import csv
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pybullet as p
import pybullet_data

# Function to read data from CSV
def readData():
    inputs = []  # Inputs (positions x, y, z and orientations alpha, beta, gamma)
    outputs = []  # Outputs (joint angles q1 to q7)

    with open('panda_data2_30.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            # Extract inputs (x, y, z, alpha, beta, gamma) and outputs (q1 to q7)
            inputs.append([float(coord) for coord in row[7:]])  # Add inputs (positions, orientations)
            outputs.append([float(angle) for angle in row[:7]])  # Add outputs (joint angles)

    return np.array(inputs), np.array(outputs)

# Read and prepare data
inputs, outputs = readData()

# Normalize data
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(inputs)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(outputs)

# Define neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(6,)),   # 6 inputs (x, y, z, alpha, beta, gamma)
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(7)  # 7 outputs (q1 to q7)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32, validation_split=0.2)

# Function to calculate positions and orientations based on joint angles using PyBullet
def calculatePositionsAndOrientations(joint_angles):
    # Set up PyBullet
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load URDF model of the robotic arm
    robot_id = p.loadURDF("franka_panda/panda.urdf")

    # Set joint angles
    for i in range(7):
        p.resetJointState(robot_id, i, joint_angles[i])

    # Get end effector position and orientation
    end_effector_pos, end_effector_ori = p.getLinkState(robot_id, 7)[:2]

    # Disconnect PyBullet
    p.disconnect()

    # Return calculated positions and orientations
    return end_effector_pos, end_effector_ori

# Predict joint angles for the first 1000 points
predicted_angles_all = []
for i in range(10000):
    # Predict joint angles
    predicted_angles_scaled = model.predict(X_train_scaled[i:i+1])
    predicted_angles = scaler_y.inverse_transform(predicted_angles_scaled)[0]
    predicted_angles_all.append(predicted_angles)

# Write predicted joint angles to CSV file
with open('predicted_joint_angles_30.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7'])  # Write header
    for angles in predicted_angles_all:
        writer.writerow(angles)  # Write predicted joint angles for each point
