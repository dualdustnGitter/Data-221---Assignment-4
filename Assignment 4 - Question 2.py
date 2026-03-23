### Question 2
# Use decision tree to train
#   use 80/20 split
#   Entropy splitting criterion
#   get training accuracy/test accuracy of model
#   
# what entropy represents in context of decision trees
# are observed results suggesting overfitting or good generalizaton.
###

# importing stuff, and feature matrix/target vector set up
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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
decision_tree_classifier = DecisionTreeClassifier(criterion='entropy')
decision_tree_classifier.fit(features_train, labels_train)



# Train accuracy
predicted_labels = decision_tree_classifier.predict(features_train)
accuracy = accuracy_score(labels_train, predicted_labels)

print("Train accuracy: " + str(accuracy))

# Test accuracy
predicted_labels = decision_tree_classifier.predict(features_test)
accuracy = accuracy_score(labels_test, predicted_labels)

print("Test accuracy: " + str(accuracy))



# entropy in this context represnts the way a feature effects the way a classification is found
# a higher entropy means lower accuracy
# while a lower one means higher accuracy

# With the result of train and test accuracy it would be seen as a good generalization
# Even with the training data accuracy being 100% the test data had a accuracy of 95%
# which is considered to be excellent