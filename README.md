# panda_machine_learning

In this machine learning process, we utilize the PyBullet library for data collection, model training and testing, value prediction, and subsequent visualization of results through graphs. Each step is facilitated by individual Python scripts that sequentially handle the tasks.

1. Data Collection (data.py):
Utilizes the PyBullet library for data collection.
Data is stored in the file panda_data2_30.csv.

2. Model Training and Testing (main1.py):
Uses data from panda_data2_30.csv for training and testing the model.
Inputs are q1 to q7, outputs are x, y, z, alfa, beta, gamma.
Generates accuracy and loss graphs.

3. Training and Testing with Exchanged Inputs and Outputs (main2.py):
Exchanges inputs to x, y, z, alfa, beta, gamma and outputs to q1 to q7.
Evaluates accuracies and losses for the new inputs and outputs.

4. Prediction of q1 to q7 Values (main3.py):
Computes predicted values of q1 to q7 from the first 10,000 values of the original data.
Writes the results to predicted_joint_angles_30.csv.

5. Prediction of x, y, z Values (data_xyz.py):
Determines new predicted values of x, y, z from the predicted values of q1 to q7.
Writes these values to predicted_xyz.csv.

6. Graph Generation:
Comparison of Original and Predicted Values (graphs1.py):
Generates a graph comparing original and predicted values.

End Effector Motion Graphs for Positions x, y, z (graphs2.py):
Generates graphs of end effector motion for each position.

End Effector Position Graph in Space (graphs3.py):
Generates a graph of the end effector's position in three-dimensional space.

For each file, it's necessary to run the respective command as mentioned above. This way, data collection, model training and testing, prediction of values, and finally graph generation for visual analysis of the results can be ensured step by step.
