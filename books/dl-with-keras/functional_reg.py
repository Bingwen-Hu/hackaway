# datalink: https://archive.ics.uci.edu/ml/datasets/Air+Quality

from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


aqdf = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',', header=0)

#remove first and last 2 cols
del aqdf["Date"]
del aqdf["Time"]
del aqdf["Unnamed: 15"]
del aqdf["Unnamed: 16"]

# fill NaNs in each column with the mean value
aqdf = aqdf.fillna(aqdf.mean())
Xorig = aqdf.as_matrix()

scaler =StandardScaler()
Xscaled = scaler.fit_transform(Xorig)
# store these off for predictions with unseen data
Xmeans = scaler.mean_
Xstds = scaler.scale_
y = Xscaled[:, 3]
X = np.delete(Xscaled, 3, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)


readings = Input(shape=(12, ))
x = Dense(8, activation='relu', kernel_initializer='glorot_uniform')(readings)
benzene = Dense(1, kernel_initializer='glorot_uniform')(x)

model = Model(inputs=[readings], outputs=[benzene])
model.compile(loss='mse', optimizer='adam')

NUM_EPOCHS = 20
BATCH_SIZE = 10

history = model.fit(X_train, y_train, batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS, validation_split=0.2)


y_test_ = model.predict(X_test).flatten()
for i in range(20):
    label = (y_test[i] * Xstds[3]) + Xmeans[3]
    prediction = (y_test_[i] * Xstds[3]) + Xmeans[3]
    print("Benzene Conc. expected: {:.3f}, predicted: {:.3f}".format(label, prediction))


plt.plot(np.arange(y_test.shape[0]), (y_test * Xstds[3]) / Xmeans[3],
         color='b', label='actual')
plt.plot(np.arange(y_test_.shape[0]), (y_test_ * Xstds[3]) / Xmeans[3],
         color='r', alpha=0.5, label='predicted')
plt.xlabel('time')
plt.ylabel('C6H6 concentrations')
plt.legend(loc='best')
plt.show()