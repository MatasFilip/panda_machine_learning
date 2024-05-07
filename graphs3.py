import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Načítanie prvých 10 vzoriek z CSV súboru
data = pd.read_csv('panda.data2.30.csv', nrows=10)

# Vykreslenie grafu s menším písmom na osiach
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['x'], data['y'], data['z'], c='navy')
ax.set_xlabel('X', fontsize=8)
ax.set_ylabel('Y', fontsize=8)
ax.set_zlabel('Z', fontsize=8)
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.tick_params(axis='z', labelsize=8)
plt.title('Pozície koncového efektora v priestore')
plt.show()
