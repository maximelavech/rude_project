import numpy as np

def resample(X,y):
    """
    Data resampling procedure applied during training of the Ensemble model.

    The resampling procedure computes the number of samples Smin in the minority class,
    i.e. the least represented class in the dataset, and returns a subset of the
    data by selecting all samples in the minority class and an identical number of
    samples without replacement in the other class

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The feature matrix.
    y : array-like of shape (n_samples,). Values must be in {0,1}
        The target labels.

    Returns
    -------
    X[indices] : {array-like, sparse matrix} of shape (Smin, n_features)
        Subset of X.

    y[indices] : array-like of shape (Smin,).
        Subset of y.
    """
    n = len(y)
    ones = np.sum(y)
    zeros = n - ones
    if ones > zeros:
        selected_ones = np.random.choice(np.where(y==1)[0],replace=False,size=zeros)
        selected_zeros = np.where(y==0)[0]
    else:
        selected_ones = np.where(y==1)[0]
        selected_zeros = np.random.choice(np.where(y==0)[0],replace=False,size=ones)
    indices = np.concatenate([selected_ones,selected_zeros])
    return X[indices],y[indices]