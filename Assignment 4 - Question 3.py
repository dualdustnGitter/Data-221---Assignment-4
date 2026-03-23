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

from pandas import Series

from sklearn.datasets import load_breast_cancer
breastCancerData = load_breast_cancer()

featureMatrix = breastCancerData.data
targetVector = breastCancerData.target


# using the decision tree model
# Splitting
features_train, features_test, labels_train, labels_test = train_test_split(featureMatrix, 
                                                                            targetVector, 
                                                                            test_size=0.2,
                                                                            random_state=42)

# creating/training model
decision_tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=3) # max depth set to 3
decision_tree_classifier.fit(features_train, labels_train)


# Train accuracy
predicted_labels = decision_tree_classifier.predict(features_train)
accuracy = accuracy_score(labels_train, predicted_labels)

print("Train accuracy: " + str(accuracy))

# Test accuracy
predicted_labels = decision_tree_classifier.predict(features_test)
accuracy = accuracy_score(labels_test, predicted_labels)

print("Test accuracy: " + str(accuracy))



# getting top 5 features
# source of method used: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
overallImportances = decision_tree_classifier.feature_importances_

featureNames = breastCancerData.feature_names
featureImportanceSeries = Series(overallImportances, index=featureNames)

print("\nTop 5 features: " + str(featureImportanceSeries.nlargest(5)))



# By making the model less complex, the model overall memorizes less
# from the training data and tries to predict trends instead
# so minimizing the coplexity the right amount would be efficient

# By seeing which features are more important, it reprsents which features
# have a higher weight in predicting the classification of the model/what
# the model is trying to predict
# and with this it explains why some features have higher weight then others