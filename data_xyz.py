import pybullet as p
import pybullet_data
import csv
import math

# Function to read predicted joint angles from CSV
def read_predicted_angles(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        predicted_joint_angles = [list(map(float, row)) for row in reader]
    return predicted_joint_angles

# Read predicted joint angles from file
predicted_joint_angles = read_predicted_angles('predicted_joint_angles_30.csv')

# Main function for data collection
def collect_data():
    # Initialize PyBullet simulation
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Set path to additional PyBullet data

    # Load Panda robot model
    panda = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    # Open predicted_xyz_30.csv file for writing
    with open('predicted_xyz_30.csv', 'w', newline='') as csvfile:
        # Create a writer object for writing to CSV file
        writer = csv.writer(csvfile)
        # Write header to the file
        writer.writerow(['x', 'y', 'z'])

        # For each point with predicted joint values
        for predicted_angles in predicted_joint_angles:
            # Set the position of robot joints
            p.setJointMotorControlArray(panda, [0, 1, 2, 3, 4, 5, 6], p.POSITION_CONTROL, targetPositions=[angle*math.pi/180 for angle in predicted_angles])

            # Simulate one time step
            p.stepSimulation()

            # Get position and orientation of end effector
            link_state = p.getLinkState(panda, 7, computeForwardKinematics=True)
            pos = link_state[0]
            x, y, z = pos

            # Write x, y, z values to the file
            writer.writerow([x, y, z])

    # Disconnect from PyBullet simulation
    p.disconnect()

# Run the function for data collection
collect_data()
