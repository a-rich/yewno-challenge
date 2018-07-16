from __future__ import print_function

import numpy as np
from scipy import sparse
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

# Load document-term matrices and labels for training.
train_X = joblib.load('data/train_dtm.pkl')
train_y = np.load(open('data/train_labels.npy'))
unlabeled = joblib.load('data/unlabeled_dtm.pkl')

# Merge labeled training samples with the unlabeled samples.
train_X = sparse.vstack([train_X, unlabeled])
train_X = train_X.todense()

# Expand the label array to contain -1 values for all the unlabeled samples.
unlabeled_labels = np.full(shape=(unlabeled.shape[0], 1), fill_value=-1)
train_y = np.append(arr=train_y.reshape(-1, 1), values=unlabeled_labels, axis=0)
train_y = train_y.reshape(-1,)

# NOTE: training these graph Laplacian methods took far too long -- the
# hyper-parameter space was not searched adequately before submitting this.

"""
        (LabelPropagation, {'n_neighbors': [3, 5, 7, 10],
                            'gamma': [0.25, 0.5, 0.75],
                            'kernel': ['knn', 'rbf']}),
        (LabelSpreading, {'alpha': [0.25, 0.5, 0.75],
                          'n_neighbors': [3, 5, 7, 10],
                          'gamma': [0.25, 0.5, 0.75],
                          'kernel': ['knn', 'rbf']})
"""

models = [
        (LabelPropagation, {'n_neighbors': [3],
                            'gamma': [0.25],
                            'kernel': ['rbf']}),
        ]

trained_models = []

# Perform grid search with cross-validation over all the potential models.
for model, params in models:

    m = model()
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

# Dump the best model to file.
joblib.dump(best_model, 'graph_Laplacian_model.pkl')
