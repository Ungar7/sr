from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax

def convolutional_dnn(channels, input_patch, output_patch):
    ''' Builds a CNN model for super-resolution'''
    model = Sequential()
    model.add(ZeroPadding2D((4, 4), input_shape=(channels, input_patch, input_patch)))
    model.add(Convolution2D(64, 9, 9, border_mode='same', activation='relu'))

    #model.add(ZeroPadding2D((2, 2)))
    model.add(Convolution2D(32, 1, 1, activation='relu'))

    model.add(ZeroPadding2D((2, 2)))
    model.add(Convolution2D(1, 5, 5, activation='relu'))

    #model.add(ZeroPadding2D((1, 1)))
    #model.add(Activation('relu'))
    #model.add(Convolution2D(channels, 3, 3))

    model.add(Flatten())
    model.add(Dense(channels * output_patch * output_patch, activation='tanh'))
    model.add(Dense(channels * output_patch * output_patch, activation='sigmoid'))

    # let's try sgd as the paper said
    #sgd = SGD(lr=0.0001, decay=0, momentum=0.9)
    #model.compile(loss='mean_squared_error', optimizer=sgd)

    model.compile(loss='mse', optimizer=Adam())

    return model

def create_model():
    return convolutional_dnn(channels=3, input_patch=32, output_patch=32)
    
