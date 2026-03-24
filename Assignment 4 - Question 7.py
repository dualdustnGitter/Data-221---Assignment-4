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

print("train shape:", X_train.shape)


# build cnn model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(16,3,padding="same", activation="relu"), # Conv2D layer with relu
    layers.MaxPool2D(), # MaxPool2D layer
])
model.add(layers.Flatten()) # Flatten
model.add(layers.Dense(128, activation="relu")) # Dense output layer
model.add(layers.Dropout(0,3)) # Dropout for "anti-" overfitting

model.summary()


# compile and train
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"] )
history = model.fit(X_train, y_train,validation_split=0.1,epochs=15, batch_size=64)




# save and predict test
model.save("mnist_model.h5")

from tensorflow.keras.models import load_model

loaded_mnist_classifier_model = load_model("mnist_model.h5")
testPredictions = loaded_mnist_classifier_model.predict(X_test)



# Confusion matrix
from sklearn.metrics import confusion_matrix

confusionMatrix = confusion_matrix(y_test, testPredictions)
print(confusionMatrix)



# misclassified images
