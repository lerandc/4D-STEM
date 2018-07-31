#Script for loading a trained Keras model to make predictions on experimental PACBEDs using softmax activation
#Author: Luis Rangel DaCosta, lerandc@umich.edu
#Last comment date: 7-31-2018

#Usage is as follows:
#python vgg16_softmax.py ID
#where ID is the target GPU device (0-3)

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
from keras import backend as K
import tensorflow as tf
import numpy as np
import scipy.io as sio
import time

def main():
    start_time = time.time()

    #establish target paths for the input data
    input_base = '/srv/home/lerandc/outputs/712_STO/'
    input_sub_folder = ['0_0/','05_0/','025_025/','1_0/','1_1/','2_0/','2_2/','3_0/']    
    result_path =  '/srv/home/lerandc/CNN/models/072418_vgg16_softmax/'

    x_train_list = []
    y_train_list = []

    sx, sy = 0, 0

    #load input data into array
    for current_folder in input_sub_folder:
        input_folder = input_base + current_folder
        input_images = [image for image in os.listdir(input_folder) if 'Sr_PACBED' in image]

        for image in input_images:
            cmp = image.split('_')
            if ('noise' in image):
                label = int(cmp[-2][:])
            else:
                label = int(cmp[-1][:-4])  

            #in this logic, I usually specified which noise level I wanted, as the files were named uniquely     
            if (('noise100' in image)):

                #load image as double float, scale it to (0,1) range, then convert back to single float
                img = np.load(input_folder + image).astype(dtype=np.float64)
                img = scale_range(img,0,1)
                img = img.astype(dtype=np.float32)

                #grab shape of image, then add zero channels to make sx x sy x 3 image shape
                #then create stack of images
                img_size = img.shape[0]
                sx, sy = img.shape[0], img.shape[1]
                new_channel = np.zeros((img_size, img_size))
                img_stack = np.dstack((img, new_channel, new_channel))
                x_train_list.append(img_stack)
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

    batch_size = 32
    # step 1, create bottle neck features
    save_bottleneck_features(x_train, y_train, batch_size, nb_train_samples,result_path)

    # step 2, train top model to interpret output from model base (ie, convolutional base in VGG16)
    epochs = 12
    batch_size = 32  #batch size should be integer divisor of number of images in training data set
    train_top_model(y_train, nb_class, max_index, epochs, batch_size, input_folder, result_path)

    # step 3, train whole model
    epochs = 50
    batch_size = 32
    fine_tune(x_train, y_train, sx, sy, max_index, epochs, batch_size, input_folder, result_path)

    print('Total computing time is: ')
    print(int((time.time() - start_time) * 100) / 100.0)


def save_bottleneck_features(x_train, y_train, batch_size, nb_train_samples,result_path):
    #creates set of bottleneck features by running input data through data generator once and saving base outputs
    #load pretrained model, excluding fully connected layers at end
    model = applications.VGG16(include_top=False, weights='imagenet')
    print('before featurewise center')
    
    #establish data generator
    datagen = ImageDataGenerator(
        featurewise_center=True,
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=1,
        vertical_flip=1,
        shear_range=0.05)
 
    #datagen needs to fit to images first to deterime some paramters for featurewise centering
    datagen.fit(x_train)
    print('made it past featurewise center')
    generator = datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=False)
    print('made it past generator')

    #get bottleneck predictions
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    print('made it past the bottleneck features')
    np.save(result_path + 'bottleneck_features_train.npy',
            bottleneck_features_train)

def train_top_model(y_train, nb_class, max_index, epochs, batch_size, input_folder, result_path):
    #load bottleneck predictions 
    train_data = np.load(result_path + 'bottleneck_features_train.npy')
    train_labels = y_train
    print(train_data.shape, train_labels.shape)

    #make top model
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(nb_class, activation='softmax'))

    #set optimizer settings and compile model
    lr = 0.001
    decay = 1e-6
    momentum = 0.9
    #optimizer = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    optimizer = optimizers.Adadelta()
    loss = 'categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    bottleneck_log = result_path + 'training_' + str(max_index) + '_bnfeature_log.csv'
    csv_logger_bnfeature = callbacks.CSVLogger(bottleneck_log)
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=3, verbose=1, mode='auto')

    #train top model on bottleneck features
    model.fit(train_data,train_labels,epochs=epochs,batch_size=batch_size,shuffle=True,
            callbacks=[csv_logger_bnfeature, earlystop],verbose=2,validation_split=0.2)

    with open(bottleneck_log, 'a') as log:
        log.write('\n')
        log.write('input images: ' + input_folder + '\n')
        log.write('batch_size:' + str(batch_size) + '\n')
        log.write('learning rate: ' + str(lr) + '\n')
        log.write('learning rate decay: ' + str(decay) + '\n')
        log.write('momentum: ' + str(momentum) + '\n')
        log.write('loss: ' + loss + '\n')

    #save top model weights
    model.save_weights(result_path + 'bottleneck_fc_model.h5')

def fine_tune(train_data, train_labels, sx, sy, max_index, epochs, batch_size, input_folder, result_path):
    print(train_data.shape, train_labels.shape)

    #load full model with imagenet weights
    model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(sx, sy, 3))
    print('Model loaded')

    #create top model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dropout(0.3))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.3))
    top_model.add(Dense(52, activation='softmax'))

    #load top model weights
    top_model.load_weights(result_path + 'bottleneck_fc_model.h5')

    #join models
    new_model = Sequential()
    for l in model.layers:
        new_model.add(l)
    new_model.add(top_model)

    #optimizer settings
    lr = 0.0001
    decay = 1e-6
    momentum = 0.9
    #optimizer = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    optimizer = optimizers.Adadelta()
    loss = 'categorical_crossentropy'
    new_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    fineture_log = result_path + 'training_' + str(max_index) + '_finetune_log.csv'
    csv_logger_finetune = callbacks.CSVLogger(fineture_log)
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')

    #create data generators for training model
    datagen = ImageDataGenerator(
        featurewise_center=True,
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=1,
        vertical_flip=1,
        shear_range=0.05)

    datagen.fit(train_data)

    generator = datagen.flow(
        train_data,
        train_labels,
        batch_size=batch_size,
        shuffle=True)

    validation_generator = datagen.flow(
        train_data,
        train_labels,
        batch_size=batch_size,
        shuffle=True)

    new_model.fit_generator(generator,epochs=epochs,steps_per_epoch=len(train_data) / 32,validation_data=validation_generator,validation_steps=(len(train_data)//5)//32,
            callbacks=[csv_logger_finetune, earlystop],verbose=2)

    with open(fineture_log, 'a') as log:
        log.write('\n')
        log.write('input images: ' + input_folder + '\n')
        log.write('batch_size:' + str(batch_size) + '\n')
        log.write('learning rate: ' + str(lr) + '\n')
        log.write('learning rate decay: ' + str(decay) + '\n')
        log.write('momentum: ' + str(momentum) + '\n')
        log.write('loss: ' + loss + '\n')

    new_model.save(result_path + 'FinalModel.h5')  # save the final model for future loading and prediction


def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input    

# step 4 make predictions using experiment results

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[1])
    main()
