### Question 2
# Use decision tree to train
#   use 80/20 split
#   Entropy splitting criterion
#   get training accuracy/test accuracy of model
#   
# what entropy represents in context of decision trees
# are observed results suggesting overfitting or good generalizaton.
###

import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_breast_cancer
breastCancerData = load_breast_cancer ()