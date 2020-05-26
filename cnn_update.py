from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.datasets import cifar10
from keras.utils import np_utils
import keras

# loads the CIFAR10 dataset
(x_train, y_train), (x_test, y_test)  = cifar10.load_data()

# Lets store the number of rows and columns
img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]

# store the shape of a single image 
input_shape = (img_rows, img_cols, 1)

# Now we one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]


#Function to add Convolutional and FC layers
def CNN_layers(layers):
    # create model
    model = Sequential()
    i=0
    model.add(Conv2D(96, (11, 11), padding = "same", input_shape = x_train.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    #Convolutional Layers
    while i<=layers:

        model.add(Conv2D(256*(i+1), (5, 5), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
        
        i+=1

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
    
    #FC Layers
    i=0
    while i<layers:
        # Softmax (for classification)
        model.add(Dense(num_classes))
        model.add(Activation("softmax"))
        i+=1
        
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    #print(model.summary())

    return model

#Function to reset all weights to zero
def weight_reset():
    rw = model.get_weights()
    rw = [[n*0 for n in m] for m in rw]
    model.set_weights(rw)


#Function to evaluate the accuracy
def model_acc(epochs):
    batch_size = 128
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)
    scores = model.evaluate(x_test, y_test, verbose=1)
    accuracy = scores[1]*100
    print()
    print('Test loss:', scores[0])
    print('Test accuracy:', accuracy)
    print()
    return accuracy

        

model = CNN_layers(1)
#print(model.summary())

accuracy = model_acc(10)

n=1
while accuracy<80:
    weight_reset()
    n = n+1
    model = CNN_layers(n)
    #print(model.summary())
    accuracy = model_acc(10*n)
    print()


print("Final Accuracy:", accuracy)
model.save("cifar10_model.h5")


fp = open("accuracy.txt", "w")
fp.write(str(final_accuracy))
fp.close()
