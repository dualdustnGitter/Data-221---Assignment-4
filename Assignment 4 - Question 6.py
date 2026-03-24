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
    layers.Conv2D(16,3,padding="same", activation="relu"), # ReLU
    layers.MaxPool2D(), # Pool
    layers.Conv2D(32,3,padding="same", activation="relu"), # ReLU
    layers.MaxPool2D(), # Pool
])
model.add(layers.Flatten()) # Flatten
model.add(layers.Dense(128, activation="relu")) # ReLU
model.add(layers.Dropout(0,3)) # Dropout for "anti-" overfitting
model.add(layers.Dense(10, activation="softmax")) # Softmax

model.summary()


# compile and train
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"] ) # Metrics is important for us, we chose accuracy
history = model.fit(X_train, y_train,validation_split=0.1,epochs=3, batch_size=64) # epochs is # of forward backwards