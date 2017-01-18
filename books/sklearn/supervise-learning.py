import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

# know your data is important
faces = fetch_olivetti_faces()
print(faces.DESCR)
print(faces.keys())
# a good habit is to check the data whether they are normalized
print(np.max(faces.data))
print(np.min(faces.data))
print(np.mean(faces.data))

# it is healthy to plot the data before any further exploration.
def print_faces(images, target, top_n):
    "set up the figure size in inches"
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1,
                        hspace=0.05, wspace=0.05)
    for i in range(top_n):
        # plot the images in a matrix of 20x20
        p = fig.add_subplot(20, 20, i+1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
        
        # label the image with the target value
        p.text(0, 14, str(target[i]))
        p.text(0, 60, str(i))


# ==================== support vector machine
from sklearn.svm import SVC

# the most important parameter of svm is the kernel function
svc_1 = SVC(kernel='linear')

# split the data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, 
                                                    test_size=0.25)

# implement cross validation
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem

def evaluate_cross_validation(clf, X, y, K):
    # create a k-fpld cross validation iterator
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), 
                                                     sem(scores)))




# ==================== navie bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# basic one
clf = Pipeline([
    ('vect', TfidfVectorizer())
    ('clf', MultinomialNB())])

# improve step by step
clf_1 =  Pipeline([
    ('vect', TfidfVectorizer())
    ('clf', MultinomialNB(alpha=0.01))])

clf_2 =  Pipeline([
    ('vect', TfidfVectorizer(token_pattern=r"some regular pattern"))
    ('clf', MultinomialNB(alpha=0.01))])


# can be replace by NLTK
def get_stop_words():
    result = set()
    for line in open('stopwords_en.txt', 'r').readlines():
        result.add(line.strip())
    return result

clf_3 =  Pipeline([
    ('vect', TfidfVectorizer(token_pattern=r"some regular pattern", 
                             stop_words=get_stop_words()))
    ('clf', MultinomialNB(alpha=0.01))])



# ==================== Evaluating the performance
from sklearn import metrics

def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    
    clf.fit(X_train, y_train)
    
    print("Accuracy on training set:")
    print(clf.score(X_train, y_train))
    print("Accuracy on testing set:")
    print(clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    
    print("CLassification Report:")
    print(metrics.classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))

# Warning: NOT executable plainly
# ==================== LabelEncoder and OneHotEncoder
from sklearn,preprocessing import LabelEncoder
enc = LabelEncoder()
label_encoder = enc.fit(X_train[:, 3]) # certain column in input data.
integers_class = label_encoder.transform(label_encoder.classes_) # just for inspect

# change the input data
t = label_encoder.transform(X_train[:, 3])
X_train[:, 3] = t

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
one_hot_encoder = enc.fit()

# First, convert classes to 0-N-1 integers using label_encoder
num_of_rows = X_train.shape[0]
t = label_encoder.transform(X_train[:, 0]).reshape(num_of_rows, 1)

# second, create a sparse matrix with three columns, each one indicating if the instance belongs to the class
new_features = one_hot_encoder.transform(t)

# add the new_features to the input data
new_X = np.concatenate([X_train, new_features.toarray()], axis=1)

# delete the origin 
new_X = np.delete(new_X, [0], 1)

# and update the label and so on......

# ==================== tree method and draw
# there are decision tree and extra tree
# the following code using IPython module and Graphviz to draw the tree
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
clf = clf.fit(X_train, y_train)

import pydot.StringIO
dot_data = StringIO.StringIO()
tree.export_graphviz(clf, out_file=dot_data, 
                     feature_name=['age', 'sex', '1st_class', '2nd_class', '3rd_class'])
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_png('pictureName')
from IPython.core.display import Image
Image(filename="pictureName")

# ==================== Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, random_state=10)
clf = clf.fit(X_train, y_train)
loo_cv(X_train, y_train, clf)   # should be imported from basic.py

# ==================== Extra Tree
# Extra tree is a forest method. It not only selects for each tree a different, random subset
# of features but also randomly selects the threshold for each decision
# Note: Extra tree also show the importance var for the predict
from sklearn import ensemble
clf_et = ensemble.ExtraTreeRegressor(n_estimators=10, compute_importances=True, random_state=10)
train_and_evaluate(clf_et, X_train, y_train)
print(sort(zip(clf_et.feature_importances_, boston.features.names), axis=0))
