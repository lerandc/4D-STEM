'''Adapt previous resnet example for MNIST data to CBED simulations
'''

from __future__ import print_function
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from Resnet import ResnetBuilder
import os
import sys
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import callbacks
from keras import optimizers
from keras.callbacks import EarlyStopping
import numpy as np
import time

start_time = time.time()

batch_size = 32
num_classes = 52
epochs = 12

# input image dimensions
img_rows, img_cols = 198, 198


# only load training data, let Keras automaticlly split it into training and testing part

input_folder = '/srv/home/lerandc/CNN/s8_ph0_Ti_pacbed_1111_npy_noise_peak_10/'
# input_folder = 'G:/jfeng_MSCdata_data/cbed/s8_0_0_Ti_pacbed_r7_npy/' # small size testing dataset with 5184 images
result_path = '/srv/home/lerandc/CNN/s8_ph0_Ti_pacbed_1111_npy_noise_peak_10/results/'
# result_path = 'G:/jfeng_MSCdata_data/cbed/s8_0_0_Ti_pacbed_r7_npy/results/'
input_images = [image for image in os.listdir(input_folder) if 'npy' in image]

x_train_list = []
y_train_list = []

sx, sy = 0, 0

for image in input_images:
    cmp = image.split('_')
    label = int(cmp[2][:-4])
    r = int(cmp[0].split('-')[1][1:])
    # r = 7
    # cmp = image.split('_')
    # cmp = cmp[3].split('.')
    # label = cmp[0]
    if r < 8:
        img = np.load(input_folder + image)
        # img_size = img.shape[0]
        sx, sy = img.shape[0], img.shape[1]
        # new_channel = np.zeros((img_size, img_size))
        # img_stack = np.dstack((img, new_channel, new_channel))
        # x_train_list.append(img_stack)
        # Ti-r7-3-0_9_30.npy
        x_train_list.append(img)
        y_train_list.append(label)

nb_train_samples = len(x_train_list)
print('Image loaded')
print('input shape: ')
print(sx, sy)
print('training number: ')
print(nb_train_samples)
nb_class = len(set(y_train_list))
x_train = np.concatenate([arr[np.newaxis] for arr in x_train_list])
y_train = to_categorical(y_train_list, num_classes=nb_class)
np.save(input_folder + 'results/y_train.npy', y_train)

# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
input_shape = (1, img_rows, img_cols)
x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255

print('x_train shape:', x_train.shape)
print('x_train type:', type(x_train))
print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

print(input_shape)
# print(x_test.shape)

# 3,4,6,3 correspond to resnet 50 according to definition in Resnet
# 3,4,23,3 corrspond to resnet 101
model = ResnetBuilder.build(input_shape, 52, 'bottleneck', [3, 4, 6, 4])


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

model.save('/srv/home/lerandc/CNN/s8_ph0_Ti_pacbed_1111_npy_noise_peak_10/results/FinalModel_resnet50.h5')

print('Total computing time is: ')
print(int((time.time() - start_time) * 100) / 100.0)