# Each image is a 28 by 28 pixel square (784 pixels total). A standard spit
# of the dataset is used to evaluate and compare models, where 60,000 images
# are used to train a model and a separate set of 10,000 images are used to test it.

import tensorflow as tf

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator

# For reproducible results.
seed = 7
numpy.random.seed(seed)

# Make sure internet connection is ON for downloading MNIST dataset for first time.
# loading dataset from either local or online repo
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 3-D array for storing train and test data as float types, for normalization
# In Keras, the layers used for two-dimensional convolutions expect pixel
# values with the dimensions [channels for Theano backend] | [width][height] | [channels for tensorflow backend].

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train/=255
X_test/=255


# one hot vector outputs for train and test data
number_of_classes = 10
Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)


# CNN Model structure defined
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), input_shape=(28,28,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flattened vector with dense connections
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Output for classification
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Data Augmentation to avoid overfitting
gen = ImageDataGenerator(rotation_range=9, width_shift_range=0.09, shear_range=0.4, height_shift_range=0.09, zoom_range=0.09)
test_gen = ImageDataGenerator()

# Batch formation process
train_generator = gen.flow(X_train, Y_train, batch_size=64)
test_generator = test_gen.flow(X_test, Y_test, batch_size=64)

# Training the model
model.fit_generator(train_generator, steps_per_epoch=60000//64, epochs=2, validation_data=test_generator, validation_steps=10000//64)

# Result Obtained
# 937/937 [==============================] - 603s 643ms/step - loss: 0.0631 - acc: 0.9803 - val_loss: 0.0380 - val_acc: 0.9877
