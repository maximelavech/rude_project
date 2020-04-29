import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import special_ortho_group

def balanced_train_test_split(X,y,size):
    """
    Function to split the data into Train and Test sets for evaluation

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The feature matrix.
    y : array-like of shape (n_samples,). Values must be in {0,1}
        The target labels.
    size : int
        The number of samples in each class for the test set.

    Returns
    -------
    *arrays : Training features, labels and Test features, labels
    """
    ind = np.concatenate([np.random.choice(np.where(y==0)[0],replace=False,size=size),
                          np.random.choice(np.where(y==1)[0],replace=False,size=size)])
    indtest = ind.reshape((-1,))
    indtrain = [i for i in range(X.shape[0]) if i not in indtest]
    return X[indtrain],y[indtrain],X[indtest],y[indtest]

class RandomDataset:
    """
    Generate an instance to sample an (unbalanced) dataset.

    Two generators are created by random sampling and random rotation. When sampling
    from the instance, we concatenate according to the defined ratio a number of samples
    from each generator with corresponding labels.

    Parameters
    ----------
    ratio : (float) number between 0 and 1
        Define the proportion of labels for one of the targets.
    features : (int)
        Number of features for matrix.
    range_mu : (tuple)
        The initial structure to build the mean vector will sample numbers within the tuple range.
    range_sigma : (tuple)
        The initial structure to build the covariance matrix will sample values in the tuple range

    Attributes
    ----------
    ratio : (float)
        Proportion for first class
    generator_class_zero : (function)
        Takes a integer as input and returns a feature matrix
    generator_class_one : (function)
        Takes a integer as input and returns a feature matrix

    Example
    -------
    num_samples = 1000
    num_features = 30
    ratio = 0.1
    dataGenerator = RandomDataset(ratio = ratio, features = num_features)
    X,y = dataGenerator(num_samples)
    """


    def __init__(self, ratio=0.1, features=2, range_mu=(-5, 5), range_sigma=(-2, 1)):
        self.ratio = 0.1
        self.generator_class_zero = self._create_generator(features, range_mu, range_sigma)
        self.generator_class_one = self._create_generator(features, range_mu, range_sigma)

    def __call__(self, n):
        """Sample a dataset with a proportion of self.ratio for each class"""
        n0 = int(n * self.ratio)
        n1 = n - n0
        X = np.concatenate([self.generator_class_zero(n0), self.generator_class_one(n1)])
        y = np.array([0] * n0 + [1] * n1)
        return X, y

    def plot(self, n):
        """Sample from our generator and plot a PCA of the data to visualize the structure"""
        X, y = self.__call__(n)
        if X.shape[1] > 2:
            pca = PCA(2)
            X = pca.fit_transform(X)
        plt.rcParams['figure.figsize'] = (5, 5)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.7)
        plt.grid(True)
        plt.title('Principal Component Analysis of the data')
        plt.show()

    def _create_generator(self, features, range_mu, range_sigma):
        """Create a data generator for the features matrix"""
        mu = self._get_mu(features,range_mu)
        sigma = self._get_sigma(features,range_sigma)

        def generator(n):
            return np.random.multivariate_normal(mu, sigma, size=n)

        return generator

    def _get_mu(self,features,range_mu):
        """Generate a random mean vector of size (features,) and applies a random rotation"""
        mu = np.random.uniform(range_mu[0], range_mu[1], size=(features,))
        rotation = special_ortho_group.rvs(features)
        return np.dot(rotation,mu)

    def _get_sigma(self, features, range_sigma):
        """Generate a random covariance matrix of size (features,features) and applies a random rotation"""
        sigma = np.random.uniform(range_sigma[0], range_sigma[1], size=(features, features))
        sigma = sigma + np.diag(np.ones(features))
        sigma = np.dot(sigma, sigma.transpose())
        rotation = special_ortho_group.rvs(features)
        sigma = np.dot(np.dot(rotation, sigma), rotation.transpose())
        return sigma


def order_by_size(num0, num1):
    """ Order values num0 and num1 """
    if num0 > num1:
        return num1, num0
    else:
        return num0, num1

def simulation_evaluate_num_estimators(num0,num1):
    """ Counts number of steps to visit all samples in num1 by sampling subsets of size num0 without replacement. """
    estimate_ = 0
    candidates = set(range(num1))
    while len(candidates)>0:
        candidates -= set(np.random.choice(num1,replace=False,size=num0))
        estimate_ += 1
    return estimate_

def transform_time(time_):
    """ Transform time_ to be either time_ if positive, or 1e9 if negative ("unlimited") """
    if time_ < 0:
        return 1e9
    else:
        return time_

def evaluate_num_estimators(labels, confidence_level = 0.95, num_simulations = 1000, max_time=3, verbose=False):
    """
    Function evaluate the number of estimators necessary for ensembling

    Here, we sample from the majority class without replacement and iterate the process (see function
    simulation_evaluate_num_estimators) for num_simulations if the time to perform num_simulations is
    inferior to max_time. We then return the confidence_level-quantile.

    Parameters
    ----------
    labels : array-like of shape (n_samples,). Values must be in {0,1}
        The target labels.
    confidence level : (float) number between 0.0 and 1.0. default_value set to 0.95.
        The alpha for the alpha-quantile return.
    num_simulations : (int) positive. default value set to 1000
        The desired number of simulations.
    max_time : (float)
        The max time allocated for simulations.
    verbose : (bool) default value set to False

    Returns
    -------
    num_estimators : (int) Estimate for the number of estimators.
    """
    num0 = (labels == 0).sum()
    num1 = (labels == 1).sum()
    num0, num1 = order_by_size(num0, num1)
    count = 0
    times = 0
    estimates = []
    while (count < num_simulations) and (times < transform_time(max_time)):
        before = time.time()
        estimates.append(simulation_evaluate_num_estimators(num0,num1))
        after = time.time()
        times += (after-before)
        count += 1
    sorted_estimates = np.sort(estimates)
    num_estimators = sorted_estimates[int(count*confidence_level)+1]
    if verbose:
        print('{} simulations in {} seconds to predict the number of weak learners.'.format(count,np.round(times,2)))
        print('{} weak learners needed at confidence level {}.'.format(num_estimators,confidence_level))
        if count<num_simulations:
            print('To increase number of simulations, increase max_time or set to -1 for unlimited computation time.')
    return num_estimators