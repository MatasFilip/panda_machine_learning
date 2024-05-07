import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the predicted_xyz_60.csv file into a DataFrame
df_predicted = pd.read_csv('predicted_xyz_30.csv')

# Select the first 10 rows
selected_predicted_data = df_predicted.head(10)

# Read the panda_data4_50.csv file into a DataFrame
df_actual = pd.read_csv('panda_data2_30.csv')

# Select the required columns (8th, 9th, and 10th columns)
selected_actual_data = df_actual.iloc[:, 7:10].head(10)

# Plot the data in a 3D graph
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')

# Extract x, y, z coordinates for predicted data
x_predicted = selected_predicted_data['x']
y_predicted = selected_predicted_data['y']
z_predicted = selected_predicted_data['z']

# Extract x, y, z coordinates for actual data
x_actual = selected_actual_data.iloc[:, 0]
y_actual = selected_actual_data.iloc[:, 1]
z_actual = selected_actual_data.iloc[:, 2]

# Plot predicted data points
ax.scatter(x_predicted, y_predicted, z_predicted, c='b', marker='o', label='Predikované dáta')

# Plot actual data points
ax.scatter(x_actual, y_actual, z_actual, c='r', marker='o', label='Pôvodné dáta')

# Connect points with a trajectory for predicted data
for i in range(9):
    ax.plot(x_predicted[i:i+2], y_predicted[i:i+2], z_predicted[i:i+2], c='b')

# Connect points with a trajectory for actual data
for i in range(9):
    ax.plot(x_actual[i:i+2], y_actual[i:i+2], z_actual[i:i+2], c='r')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Porovnanie pôvodných a predikovaných dát')
ax.legend()

# Reduce font size for axis labels
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.tick_params(axis='z', labelsize=8)

plt.tight_layout()
plt.show()
