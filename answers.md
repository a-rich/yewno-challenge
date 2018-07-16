Approach
==============================================================================

For a semi-supervised learning situation like this, there are multiple methods
that may be used to leverage the structure of the unlabeled data to improve
classifciation accuracy. I attempted three of these methods: self-training with
a random forest classifier, graph Laplacian label propagation, and clustering
with pseudo-labeling. My findings were that the graph Laplacian model performed
best (with more data). It also trained considerably faster than the
self-training approach -- the graph Laplacian model will scale well if updating
the model parameters using a warm start.

Self-training entails training a preliminary model on the limited set of
labeled samples, classifying the unlabeled samples using this model, merging
the sample corresponding with the most confident prediction into the training
set using the prediction as a pseudo-label, and then refitting the model to the
expanded training set. This method is flexible as it is a wrapper technique
that can be used on top of any traditional classification model.

Label propagation uses the graph Laplacian of the labeled and unlabeled samples
to diffuse the labels of the labeled samples out to the unlabeled samples in
their neighborhoods.

By clustering the labeled data alongside the unlabeled data, one can assign
pseudo-labels to the unlabeled samples by assuming that the majority of labeled
samples in that unlabeled sample's neighborhood share the same class. With this
dataset, in particular, clustering does not work well because the
dimensionality is too high and SVD / PCA does not capture adequate variance to
provide meaningful components.

Other common techniques that were not explored include generative modeling
(often done with Gaussian Mixture Models), transductive support-vector
machines, and co-training.

A common pitfall among all these methods is the assumption that all the
possible classes are represented by the training set. A good way to detect a
violation of this assumption is to identify unlabeled sample which are too far
from identified distributions / clusters / etc. and using our experts to
properly classify them.

(1) Is your system scalable w.r.t. size of your dataset? If not, how would
------------------------------------------------------------------------------
address the scalability (in terms of algorithms, infrastructure, or both)?
------------------------------------------------------------------------------
Would you be able to sketch Spark/MapReduce code for performing some of the
------------------------------------------------------------------------------
necessary computations?
------------------------------------------------------------------------------

If self-training is used with a large unlabeled data set, the initial
self-training process can take quite a long time. While it may be possible to
improve scalability by merging multiple high confidence predictions into the
training set at once, the risk of causing divergence from the optimal decision
boundaries is increased as any of these samples may have been classified
differently had the highest confidence sample been used alone to train the
model. If adopted as a streaming technique, refitting the model as new
unlabeled samples come in, self-training can be effective. The efficiency is
then bounded by the underlying classifcation model, most of which can be
parallelized.

Many techniques adjacent to the actual machine learning algorithms can easily
be parallelized (grid search over hyper-parameters, k-fold cross-validation,
PCA,and other data preprocessing steps). In addition, many machine learning
algorithms are themselves parallelizable as they often adhere to Kearns'
Statistical Query Model; this means that approximations of statistical
quantities can be accurately obtained by individual nodes in a compute cluster
(such as the gradients of adjacent batch processes). K-NN, SVM, Logistic
Regression, Expectation Maximization, naive Bayes, and other methods all fall
into this category. It is primarily iterative methods (such as the
self-training method described above) that can't be parallelized.

Clustering this high-dimensional data does not scale well on its own. However,
PCA has been shown to be quite scalable (see below) and clustering low
dimensional data *is* scalable. Unfortuantely, the dataset used for this
challenge does not contain any high magnitude eigenvectors, thus, very little
variance can be captured by singular value decomposition / principal component
analysis.

(2) If you were assigned additional experts, how would your strategy be
------------------------------------------------------------------------------
affected?
------------------------------------------------------------------------------

For self-training, I would set a minimum confidence threshold below which
unlabeled samples are not merged with the training set and are instead diverted
to the experts to classify before being added to the training set.

If using clustering, a transductive SVM, or a Gaussian mixture model, I would
identify unlabeled samples that lie beyond some maximum distance to a cluster /
minimum distance to a decison boundary (clustering), or closest to a margin of
separation (TSVM), or beyond two/three standard deviations from a distribution
(GMM). These identified samples would then be given to the experts to classify
and then added to the training set.

In general, assuming additional experts implies additional budgetary allowance
for their utilization, I would adjust the threshold that determines which
unlabeled samples are diverted to these experts instead of being diverted to
the training set. Also worth mentioning; if the amount of samples that require
expert labeling exceeds the budget to do so, the samples should be held out
until a reassessment of the model and methods improves their performance.

Apart from this, I would use the experts for evaluation (see part three below).

(3) In general, how would you assess the performances of your system?
------------------------------------------------------------------------------

I would advocate for holding out a subset of the unlabeled samples to use for
evaulation, as is done in this repo. The expert(s) will be asked to classify
the held out samples and the model's accuracy will be determined by comparing
its predictions for the held out samples with the expert's.

To fully answer the second question: if addtional experts are provided, then
their classifications of the samples will be combined taking the mode when
there is differentiation between classifications. In the event that there is an
equal representation of classes for a particular sample, and the model's
prediction matches one of these classes, then the model's prediction will be
assumed true.
