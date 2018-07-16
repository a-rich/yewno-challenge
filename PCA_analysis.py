from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.decomposition import TruncatedSVD

# Load document-term matrices and labels.
train_X = joblib.load('data/train_dtm.pkl').todense()
unlabeled = joblib.load('data/unlabeled_dtm.pkl').todense()
train_y = np.load(open('data/train_labels.npy'))

# Expand the label array to contain -1 values for all the unlabeled samples.
unlabeled_labels = np.full(shape=(unlabeled.shape[0], 1), fill_value=-1)
train_y = np.append(arr=train_y.reshape(-1, 1), values=unlabeled_labels, axis=0)

# Merge unlabeled and labeled data -- uncomment to plot unlabeled data also.
#data = np.vstack([train_X, unlabeled]).
data = train_X

# Decompose document-term matrix of labeled and unlabeled samples into two
# principal components.
# NOTE: SVD is not capable of capturing enough variance to produce a meaningful
# low-dimensional representation of the documents. Thus, clustering ->
# pseudo-labeling is not a viable technique.
svd = TruncatedSVD(n_components=2)
data = svd.fit_transform(data)
print(svd.explained_variance_ratio_)

df = pd.DataFrame(data=data, columns=['Component 1', 'Component 2'])
df = pd.concat([df, pd.Series(train_y.reshape(-1,)).rename('target')], axis=1)

targets = [('Foreign Trade and International Finance', 0),
           ('Crime and Law Enforcement', 1),
           ('Education', 2),
           ('Taxation', 3),
           ('Armed Forces and National Security', 4),
           ('Transportation and Public Works', 5),
           ('International Affairs', 6),
           ('Agriculture and Food', 7),
           ('Labor and Employment', 8),
           ('Health', 9),
           ('Environmental Protection', 10),
           ('Immigration', 11),
           ('Economics and Public Finance', 12),
           ('Science, Technology, Communications', 13),
           ('Energy', 14),
           ('Unknown', -1)]

colors = ['#3366CC', # 0.  light blue
          '#DC3912', # 1.  orange
          '#FF9900', # 2.  yellow
          '#109618', # 3.  bright green
          '#990099', # 4.  bright purple
          '#3B3EAC', # 5.  mid blue
          '#0099C6', # 6.  turquoise
          '#DD4477', # 7.  pink
          '#66AA00', # 8.  pastel green
          '#B82E2E', # 9.  maroon
          '#316395', # 10. dark turquoise
          '#994499', # 11. pastel purple
          '#22AA99', # 12. aqua
          '#AAAA11', # 13. greenish yellow
          '#6633CC', # 14. dark purple
          '#E67300'  # 15. dark yellow
          ]

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

for (target, _id), color in zip(targets, colors):
    indicesToKeep = df['target'] == _id
    ax.scatter(df.loc[indicesToKeep, 'Component 1']
               , df.loc[indicesToKeep, 'Component 2']
               , c = color
               , s = 50)
ax.legend([t[0] for t in targets])
ax.grid()

plt.show()
