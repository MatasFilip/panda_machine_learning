import pybullet as p
import pybullet_data
import csv
import math
import matplotlib.pyplot as plt

# Funkce pro načtení předpovězených úhlů kloubů z CSV
def read_predicted_angles(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Přeskočit záhlaví
        predicted_joint_angles = [list(map(float, row)) for row in reader]
    return predicted_joint_angles

# Načtení předpovězených úhlů kloubů ze souboru
predicted_joint_angles = read_predicted_angles('predicted_joint_angles_30.csv')

# Hlavní funkce pro sběr dat
def collect_data():
    # Inicializace simulace PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Nastavení cesty k dalším datům PyBullet

    # Načtení modelu robota Panda
    panda = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    # Seznamy pro ukládání hodnot x, y, z
    x_values = []
    y_values = []
    z_values = []

    # Pro každý bod s předpovězenými hodnotami kloubů
    for predicted_angles in predicted_joint_angles:
        # Nastavit polohu kloubů robota
        p.setJointMotorControlArray(panda, [0, 1, 2, 3, 4, 5, 6], p.POSITION_CONTROL, targetPositions=[angle*math.pi/180 for angle in predicted_angles])

        # Simulace jednoho časového kroku
        p.stepSimulation()

        # Získat pozici a orientaci koncového efektoru
        link_state = p.getLinkState(panda, 7, computeForwardKinematics=True)
        pos = link_state[0]
        x, y, z = pos

        # Přidat hodnoty x, y, z do seznamů
        x_values.append(x)
        y_values.append(y)
        z_values.append(z)

    # Odpojení od simulace PyBullet
    p.disconnect()

    # Grafické zobrazení hodnot x, y, z odděleně
    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(range(len(x_values)), x_values, label='X', c='slateblue')
    plt.xlabel('Počet bodov', fontsize=9)
    plt.ylabel('Pozícia x', fontsize=9)
    plt.title('Pohyb koncového efektoru zodpovedajúci pozícií x', fontsize=10)

    plt.subplot(3, 1, 2)
    plt.plot(range(len(y_values)), y_values, label='Y', c='slateblue')
    plt.xlabel('Počet bodov', fontsize=9)
    plt.ylabel('Pozícia y', fontsize=9)
    plt.title('Pohyb koncového efektoru zodpovedajúci pozícií y', fontsize=10)

    plt.subplot(3, 1, 3)
    plt.plot(range(len(z_values)), z_values, label='Z', c='slateblue')
    plt.xlabel('Počet bodov', fontsize=9)
    plt.ylabel('Pozícia z', fontsize=9)
    plt.title('Pohyb koncového efektoru zodpovedajúci pozícií z', fontsize=10)

    plt.tight_layout()
    plt.show()

# Spuštění funkce pro sběr dat
collect_data()
