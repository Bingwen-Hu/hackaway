"""
KNN means K-Nearest-Neighbors. 
First, we choose k nearest neighbors, and see what class they belong to because
they are LABELED at that time.
Seconnd, count the most common class. If there are two or more classes have the
same number, take some action and retry (such as discard the farthest point)
Finally we are sure to get the class of the input points.
"""



import numpy as np
from collections import Counter

def distance(v, w):
    "compute the distance between two vectors"
    vn = np.array(v)
    wn = np.array(w)
    return np.sqrt(sum(np.square(vn-wn)))



def majority_vote(labels):
    "assumes that labels are ordered from nearest to farthest"
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])
    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1]) # try again without the farthest
                                                   
# user interface
def knn_classify(k, labeled_points, new_point):
    "each labeled point should be a pair (point, label)"

    # order the points from nearest to farthest
    by_distance = sorted(labeled_points,
                         key=lambda (point, _): distance(point, new_point))

    # find the labels for the k closest
    k_nearest_labels = [label for _, label in by_distance[:k]]

    # and let them vote
    return majority_vote(k_nearest_labels)

# drawback
# 1. compute costly!
# 2. curse of dimensionality -- when it comes to high dimensionality, points are
# tend not to be close to one another at all
                         
    
    
# Special thanks to Joel Grus
