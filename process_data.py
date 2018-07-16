from __future__ import print_function
import re
import os
import sys
import json
import math
import glob
import random
import itertools
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def read_data(infile):
    """Read in the data file and create X and y lists.

       Return: X <list>, y <list>"""

    data_json = json.load(open(infile, 'r'))
    # Encode categorical string classes as numerical values
    classes = {k: i for i,k in enumerate(data_json.keys())}
    data = ((doc, classes[cls]) for cls, vals in data_json.items() for doc in vals)
    X_data, y_data = itertools.tee(data)
    X = list(zip(*X_data))[0]
    y = list(zip(*y_data))[1]

    return X, y

def clean_data(docs):
    """Apply regex to clean up documents.

       Return: docs <list>"""

    docs, num_docs = list(docs), len(docs)
    for i, doc in enumerate(list(docs)):
        new_doc = []
        for word in re.split(r'\s+', doc):
            if not re.search(r'(.)\1{2,}|[_\d\W]+', word):
                new_doc.append(word)
        docs[i] = ' '.join(new_doc)

        if i % 10000 == 0:
            print('Cleaning doc {} of {}'.format(i, num_docs))

    return docs

def partition_data(docs, labels, label_ratio=0.01, test_ratio=0.3,
        max_size=3000):
    """Shuffle the X and y data and isolate some label_ratio of the full
       dataset. This partition will constitute the labeled data for training.
       Then partition the remainder using test_ratio. This partion will
       constitute the unlabeled data while the remainder will serve as the test
       data and ground-truth labels.

       Return: train_X <list>, train_y <list>, unlabeled <list>, test_X <list>,
               test_y <list>"""

    # Shuffle data and labels
    X_y = list(zip(docs, labels))
    random.shuffle(X_y)
    X, y = zip(*X_y)

    # Limit the dataset size so training can be done in a reasonable amount of
    # time.
    X, y = X[:max_size], y[:max_size]

    # Compute indices for labeled / unlabeled / test ratios
    labeled_cutoff = int(math.ceil(len(X)*label_ratio))
    unlabeled_cutoff = len(X) - int(math.ceil((len(X)-labeled_cutoff)*test_ratio))

    # Partition data and labels into proper subsets
    train_X = X[:labeled_cutoff]
    unlabeled = X[labeled_cutoff:unlabeled_cutoff]
    test_X = X[unlabeled_cutoff:]
    train_y = np.array(y[:labeled_cutoff])
    test_y = np.array(y[unlabeled_cutoff:])

    return train_X, train_y, unlabeled, test_X, test_y

if __name__ == '__main__':
    """Given a file name for the raw document data, isolate the documents from
       their labels, clean the text documents with regex, partition the labeled
       and unlabeled data, then dump the partitioned data into their respective
       files."""

    # Accept an optional command-line argument for the data file...
    if len(sys.argv) > 1:
        infile = sys.argv[1]
    # Else select the data file from the current directory.
    else:
        files = glob.glob('*.json')
        print('Please select a data file by entering the corresponding number:')
        for i,f in enumerate(files, 1):
            print('\t{}: {}'.format(i,f))
        infile = files[input()-1]

    # Read the data in and isolate the labels.
    docs, labels = read_data(infile)

    # Clean the data using regex.
    docs = clean_data(docs)

    # Partition the data into labeled and unlabeled sets.
    train_X, train_y, unlabeled, test_X, test_y = partition_data(docs, labels)

    # Create vector space model using the labeled and unlabeled data, then
    # create document-term matrices for the different data partitions.
    vec = TfidfVectorizer(strip_accents='unicode', stop_words='english')
    vec.fit(train_X + unlabeled + test_X)
    train_dtm = vec.transform(train_X)
    unlabeled_dtm = vec.transform(unlabeled)
    test_dtm = vec.transform(test_X)

    # Create a directory for data and labels.
    if not os.path.exists('data'):
        os.makedirs('data')

    # Dump labels, vector space model, and document-term matrices to file.
    train_y.dump(open('data/train_labels.npy', 'w'))
    test_y.dump(open('data/test_labels.npy', 'w'))
    joblib.dump(vec, 'data/vector_space_model.pkl')
    joblib.dump(train_dtm, 'data/train_dtm.pkl')
    joblib.dump(unlabeled_dtm, 'data/unlabeled_dtm.pkl')
    joblib.dump(test_dtm, 'data/test_dtm.pkl')
