#Script for training Keras model from scratch to make predictions on experimental PACBEDs
#Author: Luis Rangel DaCosta, lerandc@umich.edu
#Last comment date: 7-31-2018

#Usage is as follows:
#python vgg16_scratch.py ID
#where ID is the target GPU device (0-3)

import os
import sys
import numpy as np
import scipy.io as sio
import tensorflow as tf
import time
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import callbacks
from keras.callbacks import EarlyStopping
from keras import optimizers


def main():
    start_time = time.time()

    #establish target paths for the input data
    input_base = '/srv/home/lerandc/outputs/712_STO/'
    input_sub_folder = ['0_0/','05_0/','025_025/','1_0/','1_1/','2_0/','2_2/','3_0/']    
    result_path =  '/srv/home/lerandc/CNN/models/071718_vgg16scratch/noise100/'

    x_train_list = []
    y_train_list = []

    sx, sy = 0, 0

    #load input data into array
    for current_folder in input_sub_folder:
        input_folder = input_base + current_folder
        input_images = [image for image in os.listdir(input_folder) if 'Sr_PACBED' in image]

        for image in input_images:
            #in this logic, I usually specified which noise level I wanted, as the files were named uniquely     
            if (('noise100' in image)):
                cmp = image.split('_')
                if ('noise' in image):
                    label = int(cmp[-2][:])
                else:
                    label = int(cmp[-1][:-4])  

                #load image as double float, scale it to (0,1) range, then convert back to single float
                img = np.load(input_folder + image).astype(dtype=np.float64)
                img = scale_range(img,0,1)
                img = img.astype(dtype=np.float32)

                #grab shape of image, then add zero channels to make sx x sy x 3 image shape
                #then create stack of images
                img_size = img.shape[0]
                sx, sy = img.shape[0], img.shape[1]
                img = np.reshape(img,(sx,sy,1))
                x_train_list.append(img)
                y_train_list.append(label)

    nb_train_samples = len(x_train_list)
    print('Image loaded')
    print('input shape: ')
    print(sx, sy)
    print('training number: ')
    print(nb_train_samples)
    nb_class = len(set(y_train_list))

    #creates numpy input tensor as required by keras, with shape N x sx x sy x 3
    x_train = np.concatenate([arr[np.newaxis] for arr in x_train_list])

    #performs one-hot encoding on labels, requirement for training categorical models
    y_train = to_categorical(y_train_list, num_classes=nb_class)
    print('Size of image array in bytes')
    print(x_train.nbytes)
    np.save(result_path + 'y_train.npy', y_train)


    #checks to see if model has been run before in current result folder, creates new log file if so
    logs = [log for log in os.listdir(result_path) if 'log' in log]
    max_index = 0
    for log in logs:
        cur = int(log.split('_')[1])
        if cur > max_index:
            max_index = cur
    max_index = max_index + 1

    #define input shape and load model architecture
    input_shape = (sx,sy,1)
    model = applications.VGG16(include_top=False,weights=None,input_shape=input_shape)
    
    #create and add top model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    #top_model.add(Dropout(0.3))
    top_model.add(Dense(52, activation='sigmoid'))

    final_model = Sequential()
    for layer in model.layers:
        final_model.add(layer)

    final_model.add(top_model)
    batch_size = 32

    #establish data generators
    """
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
    """

    #optimizer settings and final trainng
    lr = 0.005
    decay = 1e-6
    momentum = 0.9
    epochs = 12
    #optimizer = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    optimizer = optimizers.Adadelta(lr=1.0,rho=0.95,epsilon=None,decay=0.)
    loss = 'categorical_crossentropy'
    final_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    final_model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,shuffle=True,verbose=2,validation_split=0.2)
    #final_model.fit_generator(generator,epochs=epochs,steps_per_epoch=len(x_train) / 32,validation_data=validation_generator,validation_steps=(len(x_train)//5)//32,verbose=2)

    final_model.save(result_path + 'FinalModel.h5')  # save the final model for future loading and prediction

def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input    

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[1])
    main()