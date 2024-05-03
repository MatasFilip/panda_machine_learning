import pybullet as p
import pybullet_data
import csv
import math

# Main function for data collection
def collect_data():
    # Initialize PyBullet simulation
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Set path to additional PyBullet data

    # Load Panda robot model
    panda = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    # Open data.csv file for writing
    with open('panda.data2.30.csv', 'w', newline='') as csvfile:
        # Create a writer object for writing to CSV file
        writer = csv.writer(csvfile)
        # Write header to the file
        writer.writerow(['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'x', 'y', 'z', 'alfa', 'beta', 'gamma'])

        # For each joint with n degree step
        for q1 in range(-90, 91, 30):
            for q2 in range(-90, 91, 30):
                for q3 in range(-90, 91, 30):
                    for q4 in range(-90, 91, 30):
                        for q5 in range(-90, 91, 30):
                            for q6 in range(-90, 91, 30):
                                for q7 in range(-90, 91, 30):
                                    # Set the position of robot joints
                                    p.setJointMotorControlArray(panda, [0, 1, 2, 3, 4, 5, 6], p.POSITION_CONTROL, targetPositions=[q1*math.pi/180, q2*math.pi/180, q3*math.pi/180, q4*math.pi/180, q5*math.pi/180, q6*math.pi/180, q7*math.pi/180])

                                    # Simulate one time step
                                    p.stepSimulation()

                                    # Get position and orientation of end effector
                                    link_state = p.getLinkState(panda, 7, computeForwardKinematics=True)
                                    pos = link_state[0]
                                    orn = link_state[1]
                                    x, y, z = pos
                                    alfa, beta, gamma = p.getEulerFromQuaternion(orn)

                                    # Write values to the file
                                    writer.writerow([q1, q2, q3, q4, q5, q6, q7, x, y, z, alfa, beta, gamma])

    # Disconnect from PyBullet simulation
    p.disconnect()

# Run the function for data collection
collect_data()
