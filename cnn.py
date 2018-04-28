import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K
from matplotlib import pyplot as plt

K.set_image_dim_ordering('th')

np.random.seed(123) # for reporduciility of scripts

#Load pre-shuffled MNIST data into train and test datasets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print X_train.shape
#(60000, 28, 28): 60000 samples image are 28 by 28 pixels each

plt.imshow(X_train[0])

#Reshape input data to include depth of 1
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

print X_train.shape
#(60000, 1, 28, 28)

#Convert data type to float32 and normalize data value to the range [0,1]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print y_train.shape
#(60000,)

print y_train[:10]
#[5 0 4 1 9 2 1 3 1 4]

#Preprocess class labels
#Convert 1-D class arrays to 10-D class matrices
Y_train = np_utils.to_categorical(y_train,10)
Y_test = np_utils.to_categorical(y_test, 10)

print Y_train.shape
#(60000, 10)

#Declare Sequential model
model = Sequential()

#Declare CNN input layer
# (32,3,3) : # of convolutional filters,
             # rows in each convolutional kernal,
             # columns in each convolutional kernal
# (1,28,28): Shape of 1 same (depth, width, height)
model.add(Conv2D(32,(3,3), activation='relu',input_shape=(1,28,28)))

print model.output_shape
#(None, 32, 26, 26)

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#prevents overfitting
model.add(Dropout(0.25))

#Add fully connected Dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
