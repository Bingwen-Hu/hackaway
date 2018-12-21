# using in periodic data
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import math





NUM_EPOCHS = 10
NUM_TIMESTEPS = 20
HIDDEN_SIZE = 10
BATCH_SIZE = 96 # 24 hours (15 min intervals


# prepare data
data = np.load("LD_250.npy")
data = data.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
data = scaler.fit_transform(data)


X = np.zeros((data.shape[0], NUM_TIMESTEPS))
y = np.zeros((data.shape[0], 1))
for i in range(len(data) - NUM_TIMESTEPS - 1):
    X[i] = data[i : i+NUM_TIMESTEPS].T
    y[i] = data[i + NUM_TIMESTEPS + 1]

# reshape X to three dimensions (samples, timesteps, features)
X = np.expand_dims(X, axis=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# stateless
# model = Sequential()
# model.add(LSTM(HIDDEN_SIZE, input_shape=(NUM_TIMESTEPS, 1), return_sequences=False))
# model.add(Dense(1))
# model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
#           validation_data=(X_test, y_test), shuffle=False)

# stateful
# ensure train/test data are perfect multiples of batch_size
model = Sequential()
model.add(LSTM(HIDDEN_SIZE, stateful=True, return_sequences=False,
               batch_input_shape=(BATCH_SIZE, NUM_TIMESTEPS, 1)))
model.add(Dense(1))

# compile the model
model.compile(loss='mean_squared_error', optimizer='adam',
              metrics=['mean_squared_error'])


train_size = (X_train.shape[0] // BATCH_SIZE) * BATCH_SIZE
test_size = (X_test.shape[0] // BATCH_SIZE) * BATCH_SIZE
X_train, y_train = X_train[0:train_size], y_train[0:train_size]
X_test, y_test = X_test[0:test_size], y_test[0:test_size]
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
for i in range(NUM_EPOCHS):
    print("Epoch {:d}/{:d}".format(i+1, NUM_EPOCHS))
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=1,
              validation_data=(X_test, y_test), shuffle=False)
    model.reset_states()

score, _ = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
rmse = math.sqrt(score)
print("MSE: {:.3f}, RMSE: {:.3f}".format(score, rmse))