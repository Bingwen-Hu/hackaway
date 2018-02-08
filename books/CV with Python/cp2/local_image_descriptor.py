from PIL import Image
from scipy.ndimage import filters
import numpy as np


def compute_harris_response(img, sigma=3):
    """Compute the Harris corner detector response function
    for each pixel in a graylevel image."""
    img_x = np.zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), 0, img_x)
    img_y = np.zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), 1, img_y)

    # compute components of the Harris matrix
    W_xx = filters.gaussian_filter(img_x * img_x, sigma)
    W_yy = filters.gaussian_filter(img_y * img_y, sigma)
    W_xy = filters.gaussian_filter(img_x * img_y, sigma)

    # determinant and trace
    W_det = W_xx * W_yy - W_xy ** 2
    W_trace = W_xx + W_yy

    return W_det / W_trace

def get_harris_points(harris_img, min_dist=10, threshold=0.1):
    """Return corners form a Harris response image
    Args:
        min_dist: minimum number of pixels separating corners
        and image boundary.
    """
    corner_threshold = harris_img.max() * threshold
    harris_img_t = (harris_img > corner_threshold) * 1

    # get coordinates of candidates and values
    coords = np.array(harris_img_t.nonzero()).T
    candidate_values = [harris_img[x][y] for (x, y) in coords]

    # sort candidates
    index = np.argsort(candidate_values)

    # store allowed point locations in array
    allowed_locations = np.zeros(harris_img.shape)
    allowed_locations[min_dist: -min_dist, min_dist: -min_dist] = 1

    # select the best points taking min_distance into account
    filters_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filters_coords.append(coords[i])
            allowed_locations[(coords[i, 0]-min_dist): (coords[i, 0]+min_dist),
                              (coords[i, 1]-min_dist):(coords[i, 1]+min_dist)] = 0
    return filters_coords



def test_harris_points():
    from pylab import figure, gray, imshow, plot, axis, show
    img = Image.open("E:/Mory/rose.jpg").convert("L")
    data = np.array(img)
    harris_img = compute_harris_response(data)
    filtered_coords = get_harris_points(harris_img)
    figure()
    gray()
    imshow(data)
    plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
    axis('off')
    show()



def get_descriptors(image, filtered_coords, wid=5):
    """For each point return, pixel values around point using a
    neighbourhood of width 2*wid+1. (Assume points are extracted
    with min_distance > wid"""

    desc = []
    for coords in filtered_coords:
        patch = image[coords[0]-wid : coords[0]+wid+1,
                      coords[1]-wid : coords[1]+wid+1].flatten()
        desc.append(patch)

    return desc


def match(desc1, desc2, threshold=0.5):
    """ For each corner point descriptor in the first image, select
    its match to second image using normalized cross-correlation."""
    n = len(desc1[0])

    # pair-wise distances
    d = -np.ones((len(desc1), len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
            d2 = (desc1[j] - np.mean(desc1[j])) / np.std(desc1[j])
            ncc_value = np.sum(d1 * d2) / (n-1)
            if ncc_value > threshold:
                d[i, j] = ncc_value

    ndx = np.argsort(-d)
    matchscores = ndx[:, 0]

    return matchscores


def match_two_side(desc1, desc2, threshold=0.5):
    """ Two-sided symmetric version of match()."""
    matches_12 = match(desc1, desc2, threshold)
    matches_21 = match(desc2, desc1, threshold)

    ndx_12 = np.where(matches_12 >= 0)[0]

    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1

    return matches_12


def append_images(img1, img2):
    """ Return a new image  that appends the two images side-by-side"""

    # select the image with the fewest rows and fill in enough empty rows
    rows1 = img1.shape[0]
    rows2 = img2.shape[0]

    if rows1 < rows2:
        img1 = np.concatenate((img1, np.zeros((rows2-rows1, img1.shape[1]))), axis=0)
    elif rows1 > rows2:
        img2 = np.concatenate((img2, np.zeros((rows1-rows2, img2.shape[1]))), axis=0)
    # if none of these cases they are equal, no filling needed.

    return np.concatenate((img1, img2), axis=1)


def plot_matches(img1, img2, locates1, locates2, matchscore, show_below=True):
    """ Show a figure with lines joining the accepted matches """
