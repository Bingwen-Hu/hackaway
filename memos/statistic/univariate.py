import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns



x = np.random.randn(500)
plt.plot(x, '.')
plt.show()

# histograms
plt.hist(x, bins=25)
plt.show()

# kernel density estimation
sns.kdeplot(x)
plt.show()


# cumulative frequencies
plt.plot(stats.cumfreq(x, 25)[0])
plt.show()


# error bars
