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

# Extract values for x, y, z, alpha, beta, gamma from the selected row
x = data.iloc[5]['Position x']
y = data.iloc[5]['Position y']
z = data.iloc[5]['Position z']
alpha = data.iloc[5]['Orientation alpha']
beta = data.iloc[5]['Orientation beta']
gamma = data.iloc[5]['Orientation gamma']

# Print data from line 10
print("Data from line 5 of data.csv:")
print("x:", x)
print("y:", y)
print("z:", z)
print("alpha:", alpha)
print("beta:", beta)
print("gamma:", gamma)

# Combine position and orientation values into input tensor
input_data = torch.tensor([[x, y, z, alpha, beta, gamma]], dtype=torch.float32)

# Perform inference using the neural network to predict joint angles
with torch.no_grad():
    predicted_q = model(input_data)

# Convert predicted_q tensor to numpy array
predicted_q = predicted_q.numpy()

# Print predicted joint angles q1 to q7
print("Predicted Joint Angles (q1 to q7):", predicted_q)

# Set up PyBullet and load the robot model
p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

# Set the joint angles using predicted values
for i in range(7):
    p.resetJointState(robot, i + 1, predicted_q[0][i])

# Run the simulation for a few steps
for _ in range(100):
    p.stepSimulation()

# Get the new position and orientation of the robot's end-effector
new_position, new_orientation = p.getLinkState(robot, 11)[:2]

# Use the initial orientation values
new_alpha, new_beta, new_gamma = alpha, beta, gamma

# Now you have the new values of x, y, z, alpha, beta, gamma
print("New position:", new_position)
print("New orientation (in Euler angles):", np.degrees([new_alpha, new_beta, new_gamma]))

# Disconnect from the PyBullet physics server
p.disconnect()

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Original position
ax.scatter(x, y, z, c='r', marker='o', label='Original Position')

# New position
ax.scatter(new_position[0], new_position[1], new_position[2], c='b', marker='^', label='New Position')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()
