import h5py
import numpy as np

temperature = np.random.random(1024)
wind = np.random.random(2048)
f = h5py.File('weather.hdf5')
f['/15/temperature'] = temperature
f['/15/temperature'].attrs['dt'] = 10.0
f['/15/temperature'].attrs['start_time'] = 1239556
f['/15/wind'] = wind
f['/15/wind'].attrs['dt'] = 5.0

dataset = f['/15/temperature']
for key in dataset.attrs:
    print(f"{key}: {dataset.attrs[key]}")
