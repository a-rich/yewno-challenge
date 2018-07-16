from __future__ import print_function

import numpy as np

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

# Load document-term matrices and labels for training.
train_X = joblib.load('data/train_dtm.pkl')
train_y = np.load(open('data/train_labels.npy'))
unlabeled = joblib.load('data/unlabeled_dtm.pkl')

# Define hyper-parameters to be grid searched against each potential model.
# NOTE: these hyper-parameter values are the result of multiple iterations of
# grid searching over the parameter space.
models = [
        (LinearSVC, {'penalty': ['l2'],
                     'loss' : ['hinge', 'squared_hinge'],
                     'C': [0.19]}),
        (LogisticRegression, {'C': [1.5]}),
        (MultinomialNB, {'alpha': [0.005]}),
        (KNeighborsClassifier, {'n_neighbors': [7]}),
        (DecisionTreeClassifier, {'criterion': ['gini'],
                                  'max_depth': [2],
                                  'max_features': [0.7]}),
        (RandomForestClassifier, {'criterion': ['gini'],
                                  'n_estimators': [47],
                                  'max_depth': [3],
                                  'max_features': [0.475]}),
        (XGBClassifier, {'max_depth': [2],
                         'n_estimators': [63],
                         'gamma': [0.0005],
                         'learning_rate': [0.02]})
        ]

trained_models = []

# Perform grid search with cross-validation over all the potential models.
for model, params in models:

    if type(model()).__name__ in ['MultinomialNB', 'KNeighborsClassifier']:
        m = model()
    else:
        m = model(random_state=42)

    grid = GridSearchCV(estimator=m, param_grid=params, cv=2, n_jobs=-1)
    grid.fit(train_X, train_y)
    score = grid.score(train_X, train_y)
    print("Grid:\n", grid)
    print("Mean training time:", grid.cv_results_['mean_fit_time'])
    print("Training score:", score)
    print("Best cross-validation score:", grid.best_score_)
    print("CV/train difference:", score - grid.best_score_)
    print("Best params: {}\n".format(grid.best_params_))

    trained_models.append((grid.best_score_, grid.best_params_, grid.best_estimator_))

# Determine the model with the highest cross validation score.
best_model = sorted(trained_models, reverse=True)[0][-1]
print("Best model is {} with a cross-validation score of {}".format(
        type(best_model).__name__, sorted(trained_models, reverse=True)[0][0]))

# Dump the best model to file for self-training.
joblib.dump(best_model, 'self-training_base_model.pkl')
