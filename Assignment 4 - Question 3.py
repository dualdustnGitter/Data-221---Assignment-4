### Question 3
# Use decision train with at least one constraint
#   using max_depth, min_samples_split, or etc
# 
# report training and test accuracy
# display top 5 import features according to model
# 
# 
### 

# importing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_breast_cancer
breastCancerData = load_breast_cancer()

featureMatrix = breastCancerData.data
targetVector = breastCancerData.target