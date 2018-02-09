"""
Decision tree is fitable for quality and quanlity data
traits: easy to explain, compute costly, easy overfitting(solved Random Forest)

Entropy: message metrics
attributes: using to split the dataset

procedure:
1 compute the entropy of the whole dataset
2 foreach attribute to split the dataset
3 choose the attribute which minimize the sum of entropy of datasubset
4 for every datasubset continue split with attribute...
5 if attribute is empty, label the data the most common class in that subset


low uncertainty means low entropy

H(S) = -p1log2p1 -p2log2p2 -p3log3p3 ... -pnlog2pn
with convention that 0 log 0 = 0

"""
import numpy as np
from collections import Counter, defaultdict


def entropy(class_probabilities):
    "given a list of class probabilities, compute the entropy"
    return sum(-p * np.log2(p) for in class_probabilities if p)

# the label is the result
def class_probabilities(labels):
    total_count = len(lebels)
    return [count / total_count for count in Counter(labels).values()]

def data_entropy(labeled_data):
    "compute the entropy of given data"
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(llabels)
    return entropy(probabilities)


"""
Mathematically, if we partition our data S into subsets S1, S2 .. Sm containing 
proportions q1 q2 ... qm of the data, then we compute the entropy of the 
partition as a weighted:

H = q1H{S1} + ... + qmH(Sn)
"""

# the following approach may cause overfitness. when apply a random forest
# we can randomly choose several attributes to train our subtree
def partition_entropy(subsets):
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset) * len(subset) / total_count
               for subset in subsets)

# assume that our inputs are vector of (attribute_dict, label)
def partition_by(inputs, attribute):
    "group the data into several subsets by attribute"
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][attribute]
        groups[key].append(input)
    return groups

def partition_entropy_by(inputs, attribute):
    "computes the entropy corresponding to the given partition"
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())

def minimize_entropy_attribute(inputs, attributes): 
    pairs = [(key, partition_entropy_by(inputs, key))
              for key in attributes]
    return min(pairs, key=lambda x: x[1])

"""
our output is also a tree. Something like this:
('level',
 {'Junior': ('phd', {'no':True, 'yes': False})
  'Mid': True
  'Senior': ('tweets', {'no':False, 'yes': True})})

when we apply our model, we decide our decision tree and then predict the result
for new input
"""

def classify(tree, input):
    """ tree is our decision model and input is unknown
    every time it recurse it choose a subtree """
    # if this is a leaf node, return its value
    if tree in [True, False]:
        return tree

    attribute, subtree_dict = tree
    # see the attribute of input one by one
    subtree_key = input.get(attribute)

    if subtree_key not in subtree_dict:
        subtree_key = None

    subtree = subtree_dict[subtree_key]
    return classify(subtree, input)

    
def build_tree_id3(inputs, split_candidates=None):

    # if this is our first pass
    # all keys of the first input are split candidates
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()

    # count Trues and Falses in the inputs
    num_inputs = len(inputs)
    num_trues = len([label for _, label in inputs if label])
    num_falses = num_inputs - num_trues


    # true and false means different class
    # if only one class, stop recurse
    if num_trues == 0: return False
    if num_falses == 0: return True

    # if attribute is empty, stop recurse
    if not split_candidates:
        return num_trues >= num_falses

    # otherwise, split on the best attribute
    best_attribute = min(split_candidates,
                         key=partial(partition_entropy_by, inputs))

    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates
                      if a != best_attribute]

    # recursively build the subtrees
    subtrees = { attribute_value : build_tree_id3(subset, new_candidates)
                 for attribute_value, subset in partitions.iteritems()}
    subtrees[None] = num_trees > num_falses # default cases

    return (best_attribute, subtrees)

""" Result of the function build_tree_id3 like this
('level',
 {'Junior': ('phd', {'no':True, 'yes': False})
  'Mid': True
  'Senior': ('tweets', {'no':False, 'yes': True})})

used as the parameter to classify

tree = build_tree_id3(inputs)
classify(tree, newinput)

"""

""" Random Forest
using different training set to train trees and for every input every tree vote for it
choose the most-voted result
"""

def forest_classify(trees, input):
    votes = [classify(tree, input] for tree in trees]
    vote_counts = Counter(votes)
    return vote_counts.most_common(1)[0][0]

# In summary two way to build subtrees
#   choose a subset of data
#   choose a subset of attributes
