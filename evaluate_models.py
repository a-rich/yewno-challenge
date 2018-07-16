from __future__ import print_function

import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

# Load test data and labels for model evaluation.
test_X = joblib.load('data/test_dtm.pkl')
test_y = np.load(open('data/test_labels.npy'))

# Load base model, self-trained, and graph Laplacian models..
base_model = joblib.load('self-training_base_model.pkl')
self_trained_model = joblib.load('self-trained_model.pkl')
graph_Laplacian_model = joblib.load('graph_Laplacian_model.pkl')

# Get predictions on test data using the base, self-trained, and graph
# Laplacian models.
base_pred = base_model.predict(test_X)
self_trained_pred = self_trained_model.predict(test_X)
graph_Laplacian_pred = graph_Laplacian_model.predict(test_X.todense())

# Print accuracy of base, self-trained, and graph Laplacian models.
print('Base model accuracy:', accuracy_score(base_pred, test_y))
print('Self-trained model accuracy:', accuracy_score(self_trained_pred, test_y))
print('Graph Laplacian model accuracy:', accuracy_score(graph_Laplacian_pred, test_y))
