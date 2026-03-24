### Question 6
# using CNN 
# 
# 
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
loss, accuracy = model.evaluate(X_test, y_test)

print("Accuracy of CNN: " + accuracy)


# CNN's are meant to work on these images since it works with 2D matrices
# while normal neural networks work on 1D vectors which we need to flatten
# the image into which causes a whole dimension worth of features to be lost
# which overall decreases accuracy

# the convolution layer here is learning how to predict what actually is being shown
# by looking at small patterns which allows it to classify different things in the image