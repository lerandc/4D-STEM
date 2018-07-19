'''Adapt previous resnet example for MNIST data to CBED simulations
'''

from __future__ import print_function
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
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

def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input    

start_time = time.time()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[1])

batch_size = 32
num_classes = 52
epochs = 30

# input image dimensions
img_rows, img_cols = 157, 157


# only load training data, let Keras automaticlly split it into training and testing part

input_base = '/srv/home/lerandc/outputs/712_STO/'
input_sub_folder = ['0_0/','05_0/','025_025/','1_0/','1_1/','2_0/','2_2/','3_0/']    
result_path =  '/srv/home/lerandc/CNN/models/071818_resnet/retrain_augment_noise100/'
model_path = '/srv/home/lerandc/CNN/models/071718_resnet/noise100/'

x_train_list = []
y_train_list = []

sx, sy = 0, 0

for current_folder in input_sub_folder:
    input_folder = input_base + current_folder
    input_images = [image for image in os.listdir(input_folder) if 'Sr_PACBED' in image]

    for image in input_images:
        if (('noise100' in image)):
            cmp = image.split('_')
            if ('noise' in image):
                label = int(cmp[-2][:])
            else:
                label = int(cmp[-1][:-4])  

            img = np.load(input_folder + image).astype(dtype=np.float64)
            img = scale_range(img,0,1)
            img = img.astype(dtype=np.float32)
            #print(img)
            #print(fields)
            img_size = img.shape[0]
            sx, sy = img.shape[0], img.shape[1]
            x_train_list.append(img)
            # Ti-r7-3-0_9_30.npy
            # x_train_list.append(img)
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
np.save(result_path + 'y_train.npy', y_train)

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
model = load_model(model_path + 'FinalModel_resnet50.h5')

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

datagen = ImageDataGenerator(
    featurewise_center=True,
    rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=1,
    vertical_flip=1,
    shear_range=0.05)

datagen.fit(x_train)

generator = datagen.flow(
    x_train,
    y_train,
    batch_size=batch_size,
    shuffle=True)

validation_generator = datagen.flow(
    x_train,
    y_train,
    batch_size=batch_size,
    shuffle=True)

model.fit_generator(generator,epochs=epochs,steps_per_epoch=len(x_train) / 32,validation_data=validation_generator,validation_steps=(len(x_train)//5)//32, verbose=2)
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

model.save(result_path +'FinalModel_resnet50.h5')

print('Total computing time is: ')
print(int((time.time() - start_time) * 100) / 100.0)