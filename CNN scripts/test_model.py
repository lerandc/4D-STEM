import os
import sys
import keras
from keras.models import load_model
from keras.preprocessing import image
from keras import optimizers
import numpy as np

def main():
    #establish paths 
    input_folder = '/srv/home/lerandc/CNN/s8_ph0_Ti_pacbed_1111_npy_noise_peak_1/'
    result_path =  '/srv/home/lerandc/CNN/s8_ph0_Ti_pacbed_1111_npy_noise_peak_1/results/'
    #load a test image
    img = np.load(input_folder + 'Ti-r7-3-0_9_36.npy')
    img_size = img.shape[0]
    sx, sy = img.shape[0], img.shape[1]
    new_channel = np.zeros((img_size, img_size))
    img = np.dstack((img, new_channel, new_channel))
    img = np.reshape(img, (1, 198, 198, 3))

    #load and compile the final model 
    model = load_model(result_path + 'FinalModel.h5')
    lr = 0.005
    decay = 1e-6
    momentum = 0.9
    optimizer = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    loss = 'categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    classes = model.predict_classes(img, batch_size=10)
    print(classes)
    

    #now test resnet
    input_folder = '/srv/home/lerandc/CNN/s8_ph0_Ti_pacbed_1111_npy_noise_peak_10/'
    result_path =  '/srv/home/lerandc/CNN/s8_ph0_Ti_pacbed_1111_npy_noise_peak_10/results/'
    img = np.load(input_folder + 'Ti-r7-3-0_9_36.npy')
    img = np.reshape(img, (1, 198, 198, 1))
    
    model = load_model(result_path + 'FinalModel_resnet50.h5')
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

    classes = model.predict(img, batch_size=10)
    print(classes)





if __name__ == '__main__':
    main()