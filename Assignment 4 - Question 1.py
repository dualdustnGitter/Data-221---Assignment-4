### Question 1
# 
# 
# 
###

# importing
import numpy
import pandas
from sklearn.datasets import load_breast_cancer
breastCancerData = load_breast_cancer ()

# feature matrix and target vector
featureMatrix = breastCancerData.data # X
targetVector = breastCancerData.target # y


# getting the shape
featureMatrixShape = featureMatrix.shape
targetVectorShape = targetVector.shape

print("Feature shape: " + str(featureMatrixShape))
print("Target shape: " + str(targetVectorShape))


# number of samples
targetVectorCount = pandas.Series(targetVector).value_counts()

print(targetVectorCount)


# This dataset is not balanced since the proportion for class is not close to 50/50
# it being 357/212 or 63%/37%