"""
K-Means is one of unsupervised learning algorithms. 
Unsupervised learning using the data without labels, or if they are, ignore.

K-Means consist of four steps:
1 choose k points as k cluster center.
2 for every input, compute the distance from it to the k center points 
respectively. assign the input to the cluster which is nearest
3 if the new assignment is the same as the old one, stop it.
4 compute the new k center points and jump to step 2,

"""
import numpy as np
import random


def square_distance(v1, v2):
    diff = np.array(v1) - np.array(v2)
    return sum(np.square(diff))

class KMeans:

    def __init__(self, k):
        self.k = k
        self.means = None

    def classify(self, input):
        "return the index of the cluster the input considered to"
        return min(range(self.k),
                   key=lambda i: square_distance(input, self.mean[i]))

    def train(self, inputs):
        self.means = random.sample(inputs, self.k)
        assignments = None

        while True:
            # find new assignments
            new_assignments = map(self.classify, inputs)

            # if no assignments have changed, we're done
            if assignments == new_assignments:
                return

            # otherwise keep the new assignments
            assignments = new_assignments

            # and compute new means based on the new assignments
            for i in range(self.k):
                # find all the points assigend to cluster i
                i_points = [p for p, a in zip(inputs, assignments) if a == i]

                # compute the new k center points
                if i_points:
                    self.means[i] = np.means(i_points, axis=0) # 0 means flatten
                    
            
"""
K-Means is computed costly
And how can we choose a k?

Given a cluster, we can sum the distance of every point belong to the means
and try to find a k that minimize the total distance of all the cluster
"""

def squared_clustering_errors(inputs, k):
    "find the total squared error from k-means clustering the inputs"
    clusterer = KMeans(k)
    clusterer.train(inputs)
    means = clusterer.means
    assignments = map(clusterer.classify, inputs)

    return sum(square_distance(input, means[cluster])
               for input, cluster in zip(inputs, assignments))


