#!/usr/bin/env python
# coding: utf-8
 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.datasets import cifar10
from keras.utils import np_utils
import keras

# loads the MNIST dataset
(x_train, y_train), (x_test, y_test)  = cifar10.load_data()

# Lets store the number of rows and columns
img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]

input_shape = (img_rows, img_cols, 1)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]



def CNN_layers():
    # create model
    model = Sequential()

    # 2 sets of CRP (Convolution, RELU, Pooling)
    model.add(Conv2D(20, (5, 5), padding = "same", input_shape = x_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    model.add(Conv2D(50, (5, 5), padding = "same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    return model
    

model = CNN_layers()
# Fully connected layers (w/ RELU)
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

# Softmax (for classification)
model.add(Dense(num_classes))
model.add(Activation("softmax"))
           
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    

print(model.summary())


# Training Parameters
batch_size = 32
epochs = 10

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)

model.save("Cifar_10_model.h5")

# Evaluate the performance of our trained model
scores = model.evaluate(x_test, y_test, verbose=1)
accuracy = scores[1]*100
print('Test loss:', scores[0])
print('Test accuracy:', accuracy)

fp = open("accuracy.txt", "w")
fp.write(str(accuracy))
fp.close()
