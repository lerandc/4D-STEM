#script for training a VGG16 architecture model from scratch
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

    input_base = '/srv/home/lerandc/outputs/712_STO/'
    input_sub_folder = ['0_0/','05_0/','025_025/','1_0/','1_1/','2_0/','2_2/','3_0/']    
    result_path =  '/srv/home/lerandc/CNN/models/071718_vgg16scratch/noise100/'

    x_train_list = []
    y_train_list = []

    sx, sy = 0, 0

    a = True

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
    x_train = np.concatenate([arr[np.newaxis] for arr in x_train_list])
    y_train = to_categorical(y_train_list, num_classes=nb_class)
    print('Size of image array in bytes')
    print(x_train.nbytes)
    np.save(result_path + 'y_train.npy', y_train)


    logs = [log for log in os.listdir(result_path) if 'log' in log]
    max_index = 0
    for log in logs:
        cur = int(log.split('_')[1])
        if cur > max_index:
            max_index = cur
    max_index = max_index + 1

    input_shape = (sx,sy,1)
    model = applications.VGG16(include_top=False,weights=None,input_shape=input_shape)
    
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