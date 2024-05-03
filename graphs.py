import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the predicted_xyz_60.csv file into a DataFrame
df_predicted = pd.read_csv('predicted_xyz_30.csv')

# Select the first 5 rows
selected_predicted_data = df_predicted.head(5)

# Plot the selected predicted data in a 3D graph
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121, projection='3d')

# Extract x, y, z coordinates for predicted data
x_predicted = selected_predicted_data['x']
y_predicted = selected_predicted_data['y']
z_predicted = selected_predicted_data['z']

# Plot predicted data points
ax1.scatter(x_predicted, y_predicted, z_predicted, c='b', marker='o')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Predikované dáta')

# Reduce font size for axis labels
ax1.tick_params(axis='x', labelsize=8)
ax1.tick_params(axis='y', labelsize=8)
ax1.tick_params(axis='z', labelsize=8)

# Connect points with a trajectory
for i in range(4):
    ax1.plot(x_predicted[i:i+2], y_predicted[i:i+2], z_predicted[i:i+2], c='b')

# Read the panda_data4_50.csv file into a DataFrame
df_actual = pd.read_csv('panda_data2_30.csv')

# Select the required columns (8th, 9th, and 10th columns)
selected_actual_data = df_actual.iloc[:, 7:10].head(5)

# Plot the selected actual data in a 3D graph
ax2 = fig.add_subplot(122, projection='3d')

# Extract x, y, z coordinates for actual data
x_actual = selected_actual_data.iloc[:, 0]
y_actual = selected_actual_data.iloc[:, 1]
z_actual = selected_actual_data.iloc[:, 2]

# Plot actual data points
ax2.scatter(x_actual, y_actual, z_actual, c='r', marker='o')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Pôvodné dáta')

# Reduce font size for axis labels
ax2.tick_params(axis='x', labelsize=8)
ax2.tick_params(axis='y', labelsize=8)
ax2.tick_params(axis='z', labelsize=8)

# Connect points with a trajectory
for i in range(4):
    ax2.plot(x_actual[i:i+2], y_actual[i:i+2], z_actual[i:i+2], c='r')

plt.tight_layout()
plt.show()
