import os
import sys
import keras
from keras.models import load_model
from keras.preprocessing import image
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import numpy as np
import scipy.io as sio
import scipy.misc as sm
from PIL import Image

def main():
    #establish paths 
    #load a test imag

    input_base = '/srv/home/lerandc/Experimental Sr PACBED/S8-20180717/S8/'
    input_sub_folder = [''] #['Sr_30pm/','Sr_40pm/','Sr_50pm/']    
    result_path =  '/srv/home/lerandc/CNN/models/071618_ind_noises/noiseless/'

    x_train_list = []
    y_train_list = []
    tilt = []
    int_radius = []
    noise_level = []

    sx, sy = 0, 0

    for current_folder in input_sub_folder:
        input_folder = input_base + current_folder
        input_images = [image for image in os.listdir(input_folder) if 'mat' in image]

        for image in input_images:
            if ('SrPACBED_Stack_R8' in image):
                #print(image)
                #cmp = image.split('_')
                #print(cmp)
                #print(cmp[-1][:-4])
                label = int(19)
                #label = int( int(cmp[-1][:-4])/10 + 0.5 )
                #int_radius.append(current_folder)
                """
                if ('noise' in image):
                    label = int(cmp[-2][:])
                    int_radius.append(int(cmp[-3][:]))
                    tilt.append(current_folder)
                    noise_level.append(int(cmp[-1][5:-4]))
                else:
                    label = int(cmp[-1][:-4])   
                    int_radius.append(int(cmp[-2][:]))
                    tilt.append(current_folder)
                    noise_level.append(0) 
                """    
                #r = int(cmp[0].split('-')[1][1:])
                #if r < 8:

                img_array = sio.loadmat(input_folder + image)
                fields = sio.whosmat(input_folder + image)
                img_array = img_array[fields[0][0]]
                img_array = img_array.astype('double')
                for i in range(32):
                    img = img_array[:,:,i]
                    #img = np.load(input_folder+image)
                    img = sm.imresize(np.squeeze(img),(157, 157))
                    img = img.astype('double')
                    #print(img.dtype)
                    img = scale_range(img, 0, 1)
                    #print(img)
                    #print(fields)
                    img_size = img.shape[0]
                    sx, sy = img.shape[0], img.shape[1]
                    new_channel = np.zeros((img_size, img_size))
                    img_stack = np.dstack((img, new_channel, new_channel))
                    x_train_list.append(img_stack)
                    # Ti-r7-3-0_9_30.npy
                    # x_train_list.append(img)
                    y_train_list.append(label)
    #load and compile the final model 

    # compile setting:
    model = load_model(result_path + 'FinalModel.h5')
    lr = 0.005
    decay = 1e-6
    momentum = 0.9
    optimizer = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    loss = 'categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    p_classes = model.predict_classes(np.asarray(x_train_list), batch_size=32)
    p_arrays = model.predict(np.asarray(x_train_list), batch_size=32)
    y_list = np.asarray(y_train_list)
    p_class_list = np.asarray(p_classes)
    print(p_class_list)
    #tilt = np.asarray(tilt)
    int_radius = np.asarray(int_radius)
    #sio.savemat('exp_results2.mat',{'ylist':y_list,'p_class_list':p_class_list,'p_arrays':p_arrays})#,'int_radius':int_radius})#,'tilt':tilt,'int_radius':int_radius,'noise_level':noise_level})
    
    
    """
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

    
    datagen = image.ImageDataGenerator(
        featurewise_center=True,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1
        )
    datagen.fit(img);
    i = 0
    for batch in datagen.flow(img, batch_size=32,save_to_dir='preview',save_prefix='test',save_format='png'):
        i += 1
        if i > 20:
            break
    """

def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input    




if __name__ == '__main__':
    main()


