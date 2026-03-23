### Question 1
# make feature matrix, and target vector
# Get shape of the 2
# get number size of each class
# 
# Is dataset balanced or imbalanced
# Why are balanced datasets important 
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

# A balanced dataset is preferred for a classification model since by have a close to 50/50 split of each class
# it allows the model to more accurately predict new data fed into it, but with a class with a higher sample size
# causes a bias where its more "accurate" to another class vs the other