from PIL import Image
import numpy as np
import pylab as pl

def img_resize(img, size):
    img_ = Image.fromarray(np.uint8(img))
    return np.array(img_.resize(size))


def histeq(img, nb_bins=256):
    """ Histogram equalization of a grayscale image."""

    # get image histogram
    img_hist, bins = pl.histogram(img.flatten(), nb_bins, normed=True)
    cdf = img_hist.cumsum()     # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]   # normalize

    # use linear interpolation of cdf to find new pixel values
    img2 = pl.interp(img.flatten(), bins[:-1], cdf)

    return img2.reshape(img.shape), cdf


def test_histeq():
    img = np.array(Image.open("E:/Mory/rose.jpg").convert("L"))
    img2, cdf = histeq(img)
    pl.imshow(img2)
    pl.show()


def compute_average(img_paths):
    """ Compute the average of a list of images"""
    # open first image and make into array of type float
    average_img = np.array(Image.open(img_paths[0]), dtype=np.float32)

    for img_path in img_paths:
        try:
            average_img += np.array(Image.open(img_path))
        except:
            print(f"Skip {img_path}")
    average_img /= len(img_paths)
    return np.array(average_img, dtype=np.uint8)


def pca(X):
    """Pricipal component analysis
    Args:
        X: matrix with training data stored as flattened arrays in rows

    Returns:
        projection matrix (with import dimension first), variance, and mean
    """
    nb_data, nb_features = X.shape
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if nb_features > nb_data:
        # PCA - compact truck used
        M = np.dot(X, X.T)          # covariance matrix
        e, EV = np.linalg.eigh(M)   # eigenvalues and eigenvectors
        tmp = np.dot(X.T, EV).T     # this is the compact trick
        V = tmp[::-1]               # reverse since last eigenvectors are ones we want
        S = np.sqrt(e)[::-1]        # reverse since last eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:, i] /= S
    else:
        # PCA - SVD used
        U, S, V = np.linalg.svd(X)
        V = V[:nb_data]             # only makes sense to return the first nb_data
    return V, S, mean_X