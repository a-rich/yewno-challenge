from __future__ import print_function

import numpy as np
from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

# Load document-term matrices and labels for training.
train_X = joblib.load('data/train_dtm.pkl')
train_y = np.load(open('data/train_labels.npy'))
unlabeled = joblib.load('data/unlabeled_dtm.pkl')

# The SVM with a linear kernel is, by far, the best performing
# model...conveniently, it is very fast to train (3rd fastest after naive Bayes
# and KNN) -- KNN does not scale well with high dimensional data (we have over
# 50,000 dimensions in our setup), so we can rule it out. As naive Bayes is not
# performing well with the small number of labeled examples, the SVM is the
# optimal model to be used for self-training.
# NOTE: The above is true for my initial attempt with the full dataset, but
# training took far too long -- as a result, the best performing model after
# reducing the training set is instead a random forest.
# NOTE: Had to reduce the training set size again -- now MultinomialNB is best.
best_model = joblib.load('self-training_base_model.pkl')
params = best_model.get_params()

# Get the predicted probabilites for each class for each unlabeled sample.
predictions = best_model.predict_proba(unlabeled)

def delete_row_csr(mat, i):
    """Helper function to remove row from unlabeled document-term matrix."""

    if not isinstance(mat, sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = mat.indptr[i+1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0]-1, mat._shape[1])

# Now we implement self-training by predicting the class for each unlabeled
# sample, tranferring the most confident sample from the unlabeled set to the
# training set, and then refitting the MultinomialNB.
while unlabeled.shape[0]:
    max_index = predictions.max(axis=1).argmax(axis=0)
    best_pred = predictions[max_index]
    max_class = np.array(best_pred.argmax(axis=0)).reshape(1,)
    train_X = sparse.vstack([train_X, unlabeled[max_index]])
    train_y = np.append(train_y, max_class, axis=0)
    delete_row_csr(unlabeled, max_index)
    model = MultinomialNB(**params)
    model.fit(train_X, train_y)
    try:
        predictions = model.predict_proba(unlabeled)
    except:
        break

    if unlabeled.shape[0] % 250 == 0:
        print(unlabeled.shape[0], 'unlabeled samples remaining')

# Dump the final model to file.
joblib.dump(best_model, 'self-trained_model.pkl')
