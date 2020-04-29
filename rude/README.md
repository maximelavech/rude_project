# RUDE : A classification method for unbalanced data

#### The problem 

Consider the following problem: the mayor of a city wants to administer some medical drug to his citizens with respect 
to a given disease based on people's clinical history. The drug will have no negative effect on healthy people and could 
however potentially save a sick person. There are only a few clinically observed cases of sick people and many healthy 
people. Observing the current clinical history of the healthy and sick patients, the mayor will take no chance and wants 
to minimize the number of sick people which are classified as healthy (<i>i.e.</i> type I error). 

Mathematically the above problem can be formulated as follows: let X &#8712; R<sup>n &times; p</sup> be the feature matrix 
and y &#8712; {0,1}<sup>n</sup> be the corresponding target labels. We assume the following the scenario: we want to control the 
false positive rate, <i>i.e.</i> P(&Phi;(X) = 1 | Y = 0), where &Phi; : R<sup>p</sup> &#8594; {0,1} is an 
estimator and (X,Y) are the random variables from which the data is sampled from. We further assume that the number of 
observed true positives by far exceeds the number of true negatives, <i>i.e.</i> \#{i | y<sub>i</sub> = 0} &#8810; \#{i | y<sub>i</sub> = 1\}. 
Therefore, we have a binary classification task in an unbalanced dataset where the main goal is control to the false 
positive rate while maximizing the accuracy. Note that the negatives correspond to the minority class. 

#### Resampling Unbalanced Data for Ensembling (RUDE)

Assume the data \{(x<sub>i</sub>, y<sub>i</sub>)}<sub>i=1...n</sub> was split into a balanced test set for metric reliability and the rest of the 
unbalanced data was dumped into the training set. We let be any binary classifier and outline below our training 
strategy

1. Split training data into classes C<sub>0</sub> and C<sub>1</sub>.
2. Let n<sub>0</sub> and n<sub>1</sub> be respectively the cardinality of sets C<sub>0</sub> and C<sub>1</sub>.
3. Sample n<sub>0</sub> elements without replacement from set C<sub>1</sub> and call this new set S<sub>1</sub>.
4. Train an estimator f<sub>i</sub> on set (C<sub>0</sub> &#8746; S<sub>1</sub>).
5. Repeat Steps (1 to 4) M times. The number M can be estimated with  ```evaluate_num_estimators```.
6. The final estimator is given by &Phi;(x) = &#8721;<sub>i</sub> &Iota;(f<sub>i</sub>(x)>0.5) / M

We observe that this strategy shows very low false positive rates and substantially high accuracy. 
We believe the FPR control is mainly due to the fact that every estimator has a somewhat clear idea of the distribution 
of the minority class with respect to a subsample of the majority class.

#### Installation

```bash
pip install rude
```

#### Examples

##### Generic Example
```python
from rude import Ensemble, RandomDataset 
from rude.utils import balanced_train_test_split

dataGenerator = RandomDataset()
X, y = dataGenerator(1000)
Xtrain, ytrain, Xtest, ytest = balanced_train_test_split(X, y, size=50)

model = Ensemble()
model.fit(Xtrain, ytrain)
prediction = model.predict(Xtest)
```

##### Ensembling with a predetermined base estimator
```python
from rude import Ensemble, RandomDataset 
from rude.utils import balanced_train_test_split
# Example of base_estimator
from sklearn.gaussian_process import GaussianProcessClassifier

dataGenerator = RandomDataset()
X, y = dataGenerator(1000)
Xtrain, ytrain, Xtest, ytest = balanced_train_test_split(X, y, size=50)

model = Ensemble(GaussianProcessClassifier(kernel=None))
model.fit(Xtrain, ytrain)
prediction = model.predict(Xtest)
```
##### Evaluating the number of estimators
We can evaluate the number of estimators needed for the ensembling procedure using the function
```evaluate_num_estimators```. The number of estimators is chosen with the idea that every sample in the training set
has been observed by at least one of the estimators. Since we are using a sampling procedure, the method
just provides guarantees set by the parameter ```confidence_level```.

```python
from rude import Ensemble, RandomDataset, evaluate_num_estimators
from rude.utils import balanced_train_test_split

dataGenerator = RandomDataset()
X, y = dataGenerator(1000)
Xtrain, ytrain, Xtest, ytest = balanced_train_test_split(X, y, size=50)
M = evaluate_num_estimators(ytrain, max_time = 1)

model = Ensemble(num_estimators = M)
model.fit(Xtrain, ytrain)
prediction = model.predict(Xtest)
```

Below we provide a simple example where seeing every sample in the training set would be relevant:
![Alt text](./images/classification.png?raw=true)


