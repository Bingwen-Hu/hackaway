import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10, 0.2)
y = np.sin(x)
z = np.cos(x)

fig, axs = plt.subplots(nrows=2, ncols=1)
axs[0].plot(x, y)
axs[0].set_ylabel('Sine')

axs[1].plot(x, z)
axs[1].set_ylabel('Cosine')

plt.show()