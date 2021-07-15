import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist

import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

# Find unique numbers from the train labels

classes = np.unique(train_Y)
nClasses = len(classes)

plt.figure(figsize=[5, 5])
# view image in data set

plt.subplot(121)
plt.imshow(train_X[0, :, :], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))

plt.subplot(122)
plt.imshow(train_X[0, :, :], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))

# plt.show()

# Data pre-processing

train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)

train_X = train_X.astype('float32')
test_X = train_X.astype('float32')
train_X = train_X / 255.
test_X = train_X / 255.

# Change labels from categorical to one-hot encoding

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Display change

train_X, valid_X, train_label, valid_label = train_test_split(
        train_X, train_Y_one_hot, test_size=0.2, random_state=13)

# Training

batch_size = 64
epochs = 20
num_classes = 10

# model

fashion_model = Sequential()

fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear',
                  input_shape=(28, 28, 1), padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2), padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# train the model!

fashion_train = fashion_model.fit(
        train_X, train_label, batch_size=batch_size, epochs=epochs,
        verbose=1, validation_data=(valid_X, valid_label))
