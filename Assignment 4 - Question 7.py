### Question 7
# generate predictions on test set
# compute/display confsuoin matrix
# id and visualize at least 3 misclassified images
#   for each shopw, true label, and predicted label
# 
# one pattern observed inb miss classifications
# one realistic method to improve CNN performance
### 

from tensorflow.keras.datasets import fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# importing
import tensorflow as tf
from tensorflow.keras import layers, models 

# preprocessing Normalizing
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = X_train[..., None]
X_test = X_test[..., None]

# print("train shape:", X_train.shape)


# # build cnn model
# model = models.Sequential([
#     layers.Input(shape=(28, 28, 1)),
#     layers.Conv2D(16,3,padding="same", activation="relu"), # Conv2D layer with relu
#     layers.MaxPool2D(), # MaxPool2D layer
# ])
# model.add(layers.Flatten()) # Flatten
# model.add(layers.Dense(128, activation="relu")) # Dense output layer
# model.add(layers.Dropout(0,3)) # Dropout for "anti-" overfitting
# model.add(layers.Dense(10, activation="softmax"))

# model.summary()


# # compile and train
# model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"] )
# history = model.fit(X_train, y_train,validation_split=0.1,epochs=15, batch_size=64)




# # save and predict test
# model.save("mnist_model.h5")

import numpy
from tensorflow.keras.models import load_model

loaded_mnist_classifier_model = load_model("mnist_model.h5")
testPredictions = loaded_mnist_classifier_model.predict(X_test)

# convert prob's to class labels
testPredictions = numpy.argmax(testPredictions, axis=1)


# Confusion matrix
from sklearn.metrics import confusion_matrix

confusionMatrix = confusion_matrix(y_test, testPredictions)
print(confusionMatrix)



# misclassified images
misclassifiedImagesIndices = numpy.where(testPredictions != y_test)[0]


# show the misclassified images
import matplotlib.pyplot as plt

# source: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
for i in range(3):
    index = misclassifiedImagesIndices[i]
    
    plt.imshow(X_test[index].squeeze(), cmap="gray")
    print("True: " + str(y_test[index]))
    print("Predicted: " + str(testPredictions[index]))
    plt.show()



# One common pattern i see in the misclassifications is that the 'True" label is a 
# higher value then the predicted ones

# a way to improve the model's performance is to increase the dataset size for it 
# to train on.