import pybullet as p
import numpy as np
import pybullet_data
import csv
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def main():
    global panda  # Umožní prístup k premennej panda 
    # Štart PyBullet simulácie
    p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(1)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Načítanie robota Franka Emika Panda
    pandaStartPosition = [0, 0, 0]
    pandaStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    panda = p.loadURDF("franka_panda/panda.urdf", pandaStartPosition, pandaStartOrientation)

    # Otvorenie súboru CSV na zápis
    with open('data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # Zápis hlavičky súboru CSV
        writer.writerow(['Joint', 'Angle', 'Position x', 'Position y', 'Position z', 'Orientation alpha', 'Orientation beta', 'Orientation gamma'])

        # Zápis dát do súboru CSV a do zoznamov pre grafy
        for joint_index in range(7):  # Prechádza 7 kĺbov (q1 až q7)
            joint_name = f'q{joint_index + 1}'

            for angle in range(-90, 91, 1):
                angle_rad = np.deg2rad(angle)  # Prevod stupňov na radiány
                p.resetJointState(panda, joint_index, angle_rad)
                p.stepSimulation()
                p.setJointMotorControl2(panda, joint_index, p.POSITION_CONTROL, targetPosition=angle_rad)
                p.stepSimulation()
                end_effector_pos, end_effector_ori = p.getLinkState(panda, 11)[:2]
                
                # Získanie pozície a orientácie koncového efektora
                pos_x, pos_y, pos_z = end_effector_pos
                ori_alpha, ori_beta, ori_gamma = p.getEulerFromQuaternion(end_effector_ori)
                writer.writerow([joint_name, angle, pos_x, pos_y, pos_z, ori_alpha, ori_beta, ori_gamma])

    pos_x, pos_y, pos_z, ori_alpha, ori_beta, ori_gamma = readData()  # Načítanie údajov zo súboru CSV

    plotDataset(pos_x, pos_y, pos_z, ori_alpha, ori_beta, ori_gamma)  # Zavolanie funkcie na vytvorenie grafu

def readData():
    coordX = []  # Premenná pre ukladanie pozícií x
    coordY = []  # Premenná pre ukladanie pozícií y
    coordZ = []  # Premenná pre ukladanie pozícií z
    alpha = []   # Premenná pre ukladanie uhlov orientácie alpha
    beta = []   # Premenná pre ukladanie uhlov orientácie beta
    gamma = []   # Premenná pre ukladanie uhlov orientácie gamma

    # Načítanie údajov zo súboru CSV
    with open('data.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Preskočenie hlavičky

        for row in reader:
            joint_name, angle, pos_x, pos_y, pos_z, ori_alpha, ori_beta, ori_gamma = row
            coordX.append(float(pos_x))
            coordY.append(float(pos_y))
            coordZ.append(float(pos_z))
            alpha.append(float(ori_alpha))
            beta.append(float(ori_beta))
            gamma.append(float(ori_gamma))

    return coordX, coordY,coordZ, alpha, beta, gamma

def plotDataset(coordX, coordY,coordZ, alpha, beta, gamma):
    plt.figure(figsize=(6, 5))
    plt.scatter(coordX, alpha)
    plt.xlabel("x")
    plt.ylabel("α")
    plt.title("Pozícia x v závislosti na orientácii α koncového efektora", fontsize=10)
    plt.xticks(np.arange(-15, 6, 5))  # Nastavenie označení na osi x
    plt.yticks(np.arange(-5, 6, 5))  # Nastavenie označení na osi alpha
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.scatter(coordY, beta)
    plt.xlabel("y")
    plt.ylabel("β")
    plt.title("Pozícia y v závislosti na orientácii β koncového efektora", fontsize=10)
    plt.xticks(np.arange(-13, 2, 2))  # Nastavenie označení na osi y
    plt.yticks(np.arange(-2, 2, 1))  # Nastavenie označení na osi beta
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.scatter(coordZ, gamma)
    plt.xlabel("z")
    plt.ylabel("γ")
    plt.title("Pozícia z v závislosti na orientácii γ koncového efektora", fontsize=10)
    plt.xticks(np.arange(-150, 20, 20))  # Nastavenie označení na osi z
    plt.yticks(np.arange(-4, 5, 1))  # Nastavenie označení na osi gamma
    plt.show()

if __name__ == "__main__":
    main()


# Načítanie dát zo súboru CSV
def readData():
    inputs = []  # Vstupy (uhly klbov q1 až q7)
    outputs = []  # Výstupy (pozície x, y, z a orientácie alfa, beta, gamma)

    with open('data.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Preskočenie hlavičky
        for row in reader:
            joint_name, angle, pos_x, pos_y, pos_z, ori_alpha, ori_beta, ori_gamma = row
            inputs.append([float(angle) for angle in row[1:8]])  # Pridanie vstupov (uhly klbov)
            outputs.append([float(coord) for coord in row[2:]])  # Pridanie výstupov (pozície a orientácie)

    return np.array(inputs), np.array(outputs)

# Načítanie a príprava dát
inputs, outputs = readData()

# Rozdelenie dát na trénovaciu a testovaciu množinu
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

# Normalizácia dát
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Vytvorenie modelu neurónovej siete
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(7,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(6)  # 6 výstupov (pozície x, y, z a orientácie alfa, beta, gamma)
])

# Kompilácia modelu s metrikami accuracy a validation accuracy
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Trénovanie modelu
history = model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=32, validation_split=0.2)

model.save_weights('model_weights.weights.h5')

# Vyhodnotenie modelu na testovacích dátach
loss, acc = model.evaluate(X_test_scaled, y_test_scaled)
print("Test straty(loss):", loss)
print("Test presnosti(accuracy):", acc)

# Vykreslenie grafu modelu presnosti (accuracy)
plt.plot(history.history['accuracy'], label='Trénovanie')
plt.plot(history.history['val_accuracy'], label ='Testovanie')
plt.title('Graf modelu presnosti počas trénovania a testovania')
plt.xlabel('Epoch')
plt.ylabel('Presnosť')
plt.ylim([0, 1])
plt.legend(['Trénovanie','Testovanie'], fontsize=7, loc='lower right')
plt.show()

# Vykreslenie grafu modelu strát (loss)
plt.plot(history.history['loss'], label='Trénovanie')
plt.plot(history.history['val_loss'], label ='Testovanie')
plt.title('Graf modelu strát počas trénovania a testovania')
plt.xlabel('Epoch')
plt.ylabel('Strata')
plt.ylim([0, 1])
plt.legend(['Trénovanie','Testovanie'], fontsize=7, loc='upper right')
plt.show()

# Porovnanie skutočných hodnôt a predikcií 
predictions = model.predict(X_test)

plt.figure(figsize=(9, 6))
plt.scatter(y_test[:, 0], predictions[:, 0], label="Joint Angle 1 (q1)", c="red",alpha=0.5)
plt.title("Porovnanie skutočných hodnôt a predikcií")
plt.xlabel('Skutočná hodnota uhla [rad]')
plt.ylabel('Predpovedaná hodnota uhla [rad]')
plt.legend(loc='upper left')
actual_prediction_fig = plt.gcf()
plt.show()

plt.figure(figsize=(9, 6))
plt.scatter(y_test[:, 1], predictions[:, 1], label="Joint Angle 2 (q2)", c="blue",alpha=0.5)
plt.title("Porovnanie skutočných hodnôt a predikcií")
plt.xlabel('Skutočná hodnota uhla [rad]')
plt.ylabel('Predpovedaná hodnota uhla [rad]')
plt.legend(loc='upper left')
actual_prediction_fig = plt.gcf()
plt.show()

plt.figure(figsize=(9, 6))
plt.scatter(y_test[:, 2], predictions[:, 2], label="Joint Angle 3 (q3)", c="green",alpha=0.5)
plt.title("Porovnanie skutočných hodnôt a predikcií")
plt.xlabel('Skutočná hodnota uhla [rad]')
plt.ylabel('Predpovedaná hodnota uhla [rad]')
plt.legend(loc='upper left')
actual_prediction_fig = plt.gcf()
plt.show()

plt.figure(figsize=(9, 6))
plt.scatter(y_test[:, 3], predictions[:, 3], label="Joint Angle 4 (q4)", c="black",alpha=0.5)
plt.title("Porovnanie skutočných hodnôt a predikcií")
plt.xlabel('Skutočná hodnota uhla [rad]')
plt.ylabel('Predpovedaná hodnota uhla [rad]')
plt.legend(loc='upper left')
actual_prediction_fig = plt.gcf()
plt.show()

plt.figure(figsize=(9, 6))
plt.scatter(y_test[:, 4], predictions[:, 4], label="Joint Angle 5 (q5)", c="orange",alpha=0.5)
plt.title("Porovnanie skutočných hodnôt a predikcií")
plt.xlabel('Skutočná hodnota uhla [rad]')
plt.ylabel('Predpovedaná hodnota uhla [rad]')
plt.legend(loc='upper left')
actual_prediction_fig = plt.gcf()
plt.show()

plt.figure(figsize=(9, 6))
plt.scatter(y_test[:, 5], predictions[:, 5], label="Joint Angle 6 (q6)", c="navy",alpha=0.5)
plt.title("Porovnanie skutočných hodnôt a predikcií")
plt.xlabel('Skutočná hodnota uhla [rad]')
plt.ylabel('Predpovedaná hodnota uhla [rad]')
plt.legend(loc='upper left')
actual_prediction_fig = plt.gcf()
plt.show()

