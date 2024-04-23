import pandas as pd
import numpy as np
import pybullet as p
import pybullet_data
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define your neural network architecture
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(6, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 7)  # Output layer with 7 neurons for 7 joint angles

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the trained neural network model
model = NeuralNetwork()
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Read Data from CSV
data = pd.read_csv('data.csv')

# Select the first 5 rows from the DataFrame
selected_rows = data.head(5)

# Set up PyBullet
p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set up the plot for original positions
fig_original = plt.figure(figsize=(10, 6))
ax_original = fig_original.add_subplot(111, projection='3d')

# Set up the plot for new positions
fig_new = plt.figure(figsize=(10, 6))
ax_new = fig_new.add_subplot(111, projection='3d')

# Loop through each row in the selected data
for index, row in selected_rows.iterrows():
    # Extract values for x, y, z, alpha, beta, gamma from the current row
    x = row['Position x']
    y = row['Position y']
    z = row['Position z']
    alpha = row['Orientation alpha']
    beta = row['Orientation beta']
    gamma = row['Orientation gamma']

    # Output original position to the terminal
    print("Original Position:", (x, y, z))

    # Plot a point for the original position
    ax_original.scatter(x, y, z, color='blue')

    # Combine position and orientation values into input tensor
    input_data = torch.tensor([[x, y, z, alpha, beta, gamma]], dtype=torch.float32)

    # Perform inference using the neural network to predict joint angles
    with torch.no_grad():
        predicted_q = model(input_data)

    # Convert predicted_q tensor to numpy array
    predicted_q = predicted_q.numpy()

    # Load the robot model
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    # Set the joint angles using predicted values
    for i in range(7):
        p.resetJointState(robot, i + 1, predicted_q[0][i])

    # Run the simulation for a few steps
    for _ in range(100):
        p.stepSimulation()

    # Get the new position and orientation of the robot's end-effector
    new_position, _ = p.getLinkState(robot, 11)[:2]  # Get the position

    # Output new position to the terminal
    print("New Position:", new_position)

    # Plot a point for the new position
    ax_new.scatter(new_position[0], new_position[1], new_position[2], color='red')

# Set labels and title for original positions
ax_original.set_xlabel('X')
ax_original.set_ylabel('Y')
ax_original.set_zlabel('Z')
ax_original.set_title('Original Positions')

# Set labels and title for new positions
ax_new.set_xlabel('X')
ax_new.set_ylabel('Y')
ax_new.set_zlabel('Z')
ax_new.set_title('New Positions')

plt.show()

# Disconnect from the PyBullet physics server
p.disconnect()
