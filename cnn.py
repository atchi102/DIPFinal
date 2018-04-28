import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt

np.random.seed(123) # for reporduciility of scripts

#Load pre-shuffled MNIST data into train and test datasets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print X_train.shape
#(60000, 28, 28): 60000 samples image are 28 by 28 pixels each

plt.imshow(X_train[0])
