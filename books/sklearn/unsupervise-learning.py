# ====================  Principal Component Analysis
# PCA can reduce the dimension so we can plot the data. 
# In k-meas, visualization can help us determine the value of K

# using an example
from sklearn.datasets import load_digits
digits = load_digits()
X_digits, y_digits = digits.data, digits.target


print(digits.keys())


# visualization
import matplotlib.pyplot as plt
n_row, n_col = 2, 5

# plot max_n pictures
def print_digits(images, y, max_n=10):
    # set up the figure size in inches
    fig  = plt.figure(figsize=(2 * n_col, 2.26 * n_row))
    i = 0
    while i < max_n and i < images.shape[0]:
        p = fig.add_subplot(n_row, n_col, i + 1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone, interpolation='nearest')
        p.text(0, -1, str(y[i]))
        i += 1
    plt.show()
# print_digits(digits.images, digits.target, max_n=10)

# helper function: plot 2 pca
def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 
    'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):
        px = X_pca[:, 0][y_digits == i]
        py = X_pca[:, 1][y_digits == i]
        plt.scatter(px, py, c=colors[i])
    plt.legend(digits.target_names)
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.show()

# here. decomposition the data into 10 pca, and show two of them
from sklearn.decomposition import PCA
estimator = PCA(n_components=10)
X_pca = estimator.fit_transform(X_digits)
plot_pca_scatter()

# ============================== KMeans
import numpy as np
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(digits.data, digits.target, 
                                                                               digits.images, test_size=0.25, 
                                                                               random_state=42)
n_sample, n_features = X_train.shape
n_digits = len(np.unique(y_train))
labels = y_train

from sklearn import cluster
clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)
clf.fit(X_train)


# ==================== Affinity Propagation 
from sklearn import cluster
aff = cluster.AffinityPropagation()
aff.fit(X_train)
print(aff.cluster_centers_indices_.shape)


# ==================== MeanShift
from sklearn import cluster
ms = cluster.MeanShift()
ms.fit(X_train)
print(ms.cluster_centers_.shape)
