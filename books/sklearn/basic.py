"""
Learning scikit-learn
special thanks to the authors
"""

# sklearn basic method

# ============================== load the datasets
from sklearn import datasets
iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target
X_iris.shape, y_iris.shape

# ============================== split and scale
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
X, y = X_iris[:, :2], y_iris
X_train, X_test, y_train, y_test = train_test_split(X, y)
scalar = preprocessing.StandardScaler().fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)


# ============================== visualize
import matplotlib.pyplot as plt
colors = ['red', 'greenyellow', 'blue']
for i in range(len(colors)):
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(xs, ys, c=colors[i])
plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()


# SGD means Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(X_train, y_train)

# --============================== visualize
# set the axes
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

import numpy as np
xs = np.arange(x_min, x_max, 0.5)
fig, axes = plt.subplots(1, 3)
fig.set_size_inches(10, 6)
for i in [0, 1, 2]:
    axes[i].set_aspect('equal')
    axes[i].set_title("Class "+ str(i) + ' versus the rest')
    axes[i].set_xlabel('Sepal length')
    axes[i].set_ylabel('Sepal width')
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)
    plt.sca(axes[i])
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.prism)
    ys = (-clf.intercept_[i] - xs * clf.coef_[i, 0]) / clf.coef_[i, 1]
    plt.plot(xs, ys, hold=True)


newpoints = np.array([4.7, 3.1],
                     [2.1, 2.0])
clf.predict(scalar.transform(newpoints))

# what's the decision function?
clf.decision_function(newpoints)

# ============================== evaluate
from sklearn import metrics
y_train_pred = clf.predict(X_train)
metrics.accuracy_score(y_train, y_train_pred)
y_pred = clf.predict(X_test)
metrics.accuracy_score(y_test, y_pred)

# some metrics
# Precision: how right our classifier is when it says that an instance is pos
# Precision = TP / (TP + FP)
# Recall: how right our classifier is when faced with a positive instance
# Recall = TP / (TP + FN)
# F1-score = 2 * Precision * Recall / (Precision + Recall)

# we can get a report like this
print(metrics.classification_report(y_test, y_pred, 
                                    target_names=iris.target_names))
#              precision    recall  f1-score   support

#      setosa       1.00      0.93      0.97        15
#  versicolor       0.53      0.89      0.67         9
#   virginica       0.89      0.57      0.70        14

# avg / total       0.85      0.79      0.80        38

# confusion matrix shows the number of class instances i that were predicted 
# be in class j. (similar as FP, TP et al)
print(metrics.confusion_matrix(y_test, y_pred))


# ============================== cross validation KFold
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.pipeline import Pipeline
clf = Pipeline([('scaler', preprocessing.StandardScaler()), 
                ('linear_model', SGDClassifier())])
cv = KFold(X.shape[0], 5, shuffle=True, random_state=33)
scores = cross_val_score(clf, X, y, cv=cv)
print(scores)
# the average is the value to estimate the real value

# ============================== cross validation leave one out 
from sklearn.cross_validation import LeaveOneOut, cross_val_score
from scipy.stats import sem

def mean_score(scores):
    string = "Mean score: {0:.3f} (+/-{1:.3f})"
    return string.format(np.mean(scores), sem(scores))
print(mean_score(scores))
# Mean score: 0.800 (+/-0.038)

def loo_cv(X_train, y_train, clf):
    loo = LeaveOneOut(X_train[:].shape[0]) # number of rows
    scores = np.zeros(X_train[:].shape[0])
    for train_index, test_index in loo:
        X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
        y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
        clf.clf.fit(X_train_cv, y_train_cv)
        y_pred = clf.predict(X_test_cv)
        scores[test_index] = metrics.accuracy_score(y_test_cv.astype(int), y_pred.astype(int))
    print("Mean score: {0:.3f} {+/-{1:.3f}}".format(npp.mean(scores), sem(scores)))


