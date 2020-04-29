import numpy as np
from copy import copy
from sklearn.tree import DecisionTreeClassifier
from .sampling import resample

DEFAULT_NUMBER_OF_ESTIMATORS = 100

class Ensemble:
    """
    Ensemble Class.

    The ensembling method relies on the sampling procedure to create an ensemble of weak learners. While the
    ensembling method is straightforward, the novelty comes from the sampling procedure which was designed
    to tackle an unbalanced dataset. Indeed, during the training phase, each weak learner receives a somewhat
    different training set which consists of:
        * the entire minority class, i.e. the class with least example.
        * an equal number of examples from the majority class which are sampled without replacement.

    Attributes
    ----------
    base_estimator : (function)
        returns a copy of the base_estimator (sklearn-like BaseEstimator)
    num_estimators : (int)
        The number of estimators.
    estimators : (list)
        The list of estimators, initially empty.

    Example
    -------
    from rude import Ensemble, RandomDataset
    num_samples = 100
    dataGenerator = RandomDataset()
    X,y = dataGenerator(num_samples)
    model = Ensemble()
    model.fit(X,y)
    """

    def __init__(self, base_estimator=None, num_estimators=None):
        """__init__ method for Ensemble Class

        The __init__ method will call _load_base_estimator and _load_num_estimators methods which will return
        default estimator and value if parameters are left to None.

        Parameters
        ----------
        base_estimator : (sklearn-like BaseEstimator)  must have a fit and predict method.
            the base_estimator which will be used for training. If None the default weak leaner will
            be the sklearn.tree.DecisionTreeClassifier.
        num_estimators : (int)
            If int the number of iterations will be of num_estimators. If None the number of estimators
            will be of (DEFAULT_NUMBER_OF_ESTIMATORS = 100).
        """
        self.base_estimator = self._load_base_estimator(base_estimator)
        self.num_estimators = self._load_num_estimators(num_estimators)
        self.estimators = None

    def fit(self, X, y):
        """ Calls _fit_single_estimator and fits a copy of the base_estimator to a subset of the data.
        Note
        ----
        See _fit_single_estimator for more details
        """
        self.estimators = [self._fit_single_estimator(X, y) for _ in range(self.num_estimators)]

    def predict_score(self, X):
        """ Returns average of predictions from all weak learners. """
        return np.mean([estimator.predict(X) for estimator in self.estimators], axis=0)

    def predict(self, X):
        """ Returns class prediction from ensembling score. """
        return (self.predict_score(X) > 0.5) * 1

    def _load_num_estimators(self, num_estimators=None):
        """ Returns a integer which will be the number of weak learners for ensembling. """
        if num_estimators == None:
            return DEFAULT_NUMBER_OF_ESTIMATORS
        else:
            return num_estimators

    def _load_base_estimator(self, base_estimator=None):
        """ Returns a function which when called returns a copy of the base_estimator

        Parameters
        ----------
        base_estimator : (sklearn-like BaseEstimator)  must have a fit and predict method.
            the base_estimator which will be used for training. If None the default weak leaner will
            be the sklearn.tree.DecisionTreeClassifier.

        Returns
        -------
        return_base_estimator : (function)
            function with no arguments. returns copy of base_estimator.
        """
        if base_estimator == None:
            def return_base_estimator():
                return DecisionTreeClassifier()
        else:
            def return_base_estimator():
                return copy(base_estimator)
        return return_base_estimator

    def _fit_single_estimator(self, X, y):
        """
        Fits a weak learner to a subset of (X,y). The subset is obtained using the utils.resample function.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The feature matrix.
        y : array-like of shape (n_samples,). Values must be in {0,1}
            The target labels.

        Returns
        -------
        clf : (sklearn-like BaseEstimator)
            The fitted estimator
        """
        Xtmp, ytmp = resample(X, y)
        clf = self.base_estimator()
        clf.fit(Xtmp, ytmp)
        return clf
