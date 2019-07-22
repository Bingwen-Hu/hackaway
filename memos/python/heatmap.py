import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np


data = np.zeros((100, 100))
adder = np.abs(np.random.randn(40, 40))
center = adder / np.max(adder)
data[20:60, 20:60] = center

plt.imshow(data, cmap=cm.seismic)
plt.colorbar()
plt.show()