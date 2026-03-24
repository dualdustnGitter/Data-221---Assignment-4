### Question 5
# Confusion matrix for
#   Decision trees
#   Neural network
# 
### 

### Decision Tree
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




### Neural Networks
from tensorflow.keras.models import Sequential
neuralNetworkModel = Sequential()
from tensorflow.keras.layers import Dense, InputLayer
inputLayer = InputLayer(input_shape=(30,)) # input layer 30 neurons
# hidden layer
neuralNetworkModel.add(inputLayer) # first layer
hiddenLayer = Dense(6)
neuralNetworkModel.add(hiddenLayer)
secondHiddenlayer = Dense(5) # second
neuralNetworkModel.add(secondHiddenlayer)
# output layer
outputLayer = Dense(1, activation="sigmoid")
neuralNetworkModel.add(outputLayer)
# compile
neuralNetworkModel.compile(loss="binary_crossentropy", metrics=["accuracy"])


# import dataset
from sklearn.model_selection import train_test_split
import pandas
from sklearn.datasets import load_breast_cancer
breastCancerData = load_breast_cancer()

featureMatrix = breastCancerData.data
targetVector = breastCancerData.target
# splitting
features_train, features_test, labels_train, labels_test = train_test_split(featureMatrix, 
                                                                            targetVector, 
                                                                            test_size=0.2,
                                                                            random_state=42)
# standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

neuralNetworkModel.fit(features_train,labels_train,epochs=10)

# Predict and accuracy of Train data
newDiseaseSamples = pandas.DataFrame(features_train)
classProbabilities = neuralNetworkModel.predict(newDiseaseSamples)
# Predict and accuracy of Test data
newDiseaseSamples = pandas.DataFrame(features_test)
classProbabilities = neuralNetworkModel.predict(newDiseaseSamples)


