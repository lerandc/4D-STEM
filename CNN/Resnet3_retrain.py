#Adaptation of script for retraining Resnet model to make predictions on experimental PACBEDs
#Author: Luis Rangel DaCosta, lerandc@umich.edu, orig. script from Chenyu
#Last comment date: 7-31-2018

#Usage is as follows:
#python Resnet3_retrain.py ID
#where ID is the target GPU device (0-3)

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

#load input data into array
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

            img_size = img.shape[0]
            sx, sy = img.shape[0], img.shape[1]
            x_train_list.append(img)
            y_train_list.append(label)

nb_train_samples = len(x_train_list)
print('Image loaded')
print('input shape: ')
print(sx, sy)
print('training number: ')
print(nb_train_samples)
nb_class = len(set(y_train_list))


#creates numpy input tensor as required by keras, with shape N x sx x sy x 1
x_train = np.concatenate([arr[np.newaxis] for arr in x_train_list])

#performs one-hot encoding on labels, requirement for training categorical models
y_train = to_categorical(y_train_list, num_classes=nb_class)
np.save(result_path + 'y_train.npy', y_train)


x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
input_shape = (1, img_rows, img_cols)
x_train = x_train.astype('float32')

print('x_train shape:', x_train.shape)
print('x_train type:', type(x_train))
print(x_train.shape[0], 'train samples')
print(input_shape)

# 3,4,6,3 correspond to resnet 50 according to definition in Resnet
# 3,4,23,3 corrspond to resnet 101
model = load_model(model_path + 'FinalModel_resnet50.h5')

#compile with Adadelta optimizer
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#establish data generator
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

model.save(result_path +'FinalModel_resnet50.h5')

print('Total computing time is: ')
print(int((time.time() - start_time) * 100) / 100.0)