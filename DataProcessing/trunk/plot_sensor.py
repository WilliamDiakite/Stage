# Plot a sensor with features

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np


matplotlib.style.use('ggplot')

dataset_path = '/home/william/Documents/__Dev/PredictiveCarMaintenance/Ressources/Generated_files/Datasets/'


#-------------------------#

dataset_name = 'vasile'
normality = 'normal'
filename = '1.csv'
sensor = 'SPEED'
features = ['ma_200_SPEED', 'std_w200_SPEED', 'd1_SPEED', 'd2_SPEED']

#-------------------------#


# Init filepath 
file = dataset_path + '/' + dataset_name + '/' + normality + '/not_normalized/' + filename
raw_file = dataset_path + '/' + dataset_name + '/' + normality + '/raw/' + filename

# Read data
df = pd.read_csv(file)
raw_df = pd.read_csv(raw_file)
print(df.shape)
print(raw_df.shape)
	

# Store time series 
sensor = 'raw_' + sensor
features.append(sensor)

# Plot curves
df['Time'] = pd.Series(list(range(len(df))))
raw_df['Time'] = pd.Series(list(range(len(raw_df))))

#df[100:350].plot(x='Time', y=features, subplots=True)
'''
# Moving average
fig, axes = plt.subplots(nrows=2, ncols=1)
df['raw_SPEED'].plot(ax=axes[0], cmap="jet")
axes[0].set_title('Vitesse interpolée')
axes[0].set_xlabel('Temps ($s$)')
axes[0].set_ylabel('Vitesse ($km.h^{-1}$)')
df['ma_60_SPEED'].plot(ax=axes[1])
axes[1].set_title('Moyenne glissante (60 secondes)')
axes[1].set_xlabel('Temps ($s$)')
axes[1].set_ylabel('Vitesse ($km.h^{-1}$)')
plt.show()

# STD
fig, axes = plt.subplots(nrows=2, ncols=1)
df['raw_SPEED'].plot(ax=axes[0], cmap="jet")
axes[0].set_title('Vitesse interpolée')
axes[0].set_xlabel('Temps ($s$)')
axes[0].set_ylabel('Vitesse ($km.h^{-1}$)')
df['std_w60_SPEED'].plot(ax=axes[1])
axes[1].set_title('Ecart-type (60 secondes)')
axes[1].set_xlabel('Temps ($s$)')
axes[1].set_ylabel('Vitesse ($km.h^{-1}$)')
plt.show()

# Derivatives
fig, axes = plt.subplots(nrows=3, ncols=1)
df['raw_SPEED'].plot(ax=axes[0], cmap="jet")
axes[0].set_title('Vitesse interpolée')
axes[0].set_xlabel('Temps ($s$)')
axes[0].set_ylabel('Vitesse ($km.h^{-1}$)')

df['d1_SPEED'].plot(ax=axes[1])
axes[1].set_title('Dérivée première instantanée')
axes[1].set_xlabel('Temps ($s$)')
axes[1].set_ylabel('Accélération ($km.h^{-2}$)')

df['d2_SPEED'].plot(ax=axes[2])
axes[2].set_title('Dérivée seconde instantanée')
axes[2].set_xlabel('Temps ($s$)')
axes[2].set_ylabel(r"$\frac{dAcceleration}{dt}$ ($km.h^{-3}$)" )
plt.show()
'''
# RAW
time = raw_df['# Time'].values
points = raw_df['raw_SPEED'].values

plt.scatter(time, points, label='Données vitesse brute')
df['raw_SPEED'].plot(label='Vitesse interpolée')
plt.xlabel('Temps ($s$)')
plt.ylabel('Vitesse ($km.h^{-1}$)')
plt.legend(loc='best')
plt.show()