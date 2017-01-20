# the dataset should be download from kaggle: kaggle.com
# ==================== load the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
titanic = pd.read_csv('data/titanic.csv')

print(titanic)


# ==================== feature extraction
# the following function can encode the categorical attributes to numeric.
from sklearn import feature_extraction
def one_hot_dataframe(data, cols, replace=False):
    vec = feature_extraction.DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData)

# usage:
titanic, titanic_n = one_hot_dataframe(titanic, ['Pclass', 'Sex'.'Embarked'], replace=True)
print(titanic.describe())

# ==================== feature selection

# first, we say we have a model
from sklearn import tree
dt = tree.DecisionTreeClassifier(crtiterion='entropy')
dt = dt.fit(X_train, y_train)
from sklearn import metrics
y_pred = dt.predict(X_test)
print("Accuracy: {0:.3f}".format(metrics.accuracy_score(y_test, y_pred)), "\n")

# then we use feature selection
# feature selection can select the most relevant variable out of all the features.
from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
X_train_fs = fs.fit_transform(X_train, y_train) # new X is ready

# ==================== Grid search
# Grid search can help us tune several parameters of our model and determine the best.
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
parameters = {
    'svc__gamma': np.logspace(-2, 1, 4),
    'svc__C': np.logspace(-1, 1, 3),
}

clf = Pipeline([
    ('vect', TfidfVectorizer()),
    ('svc', SVC()),
])

gs = GridSearchCV(clf, parameters, verbose=2, refit=False, cv=3)
gs.fit(X_train, y_train)
gs.best_params_, gs.best_score_


