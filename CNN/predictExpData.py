#Script for loading a trained Keras model to make predictions on experimental PACBEDs
#Author: Luis Rangel DaCosta, lerandc@umich.edu
#Last comment date: 7-31-2018

#Usage is as follows
#in command line, call: python predictExpData.py ID model_path output_name
#where ID is 0-3 (on mscdata) for assigning a unique GPU device to be used by the script
#model_path is the folder of the desired model
#output name is the name of the matlab file you wish to write to, including ".mat"

import os
import sys
import keras
from keras.models import load_model
from keras.preprocessing import image
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import scipy.io as sio
import scipy.misc as sm
from PIL import Image

def main():
    #establish paths to data
    #often, I was dealing with data separated by folders, between these lists and some logic in the loops, I establish the full path

    input_base = '/srv/home/lerandc/Experimental Sr PACBED/'
    input_sub_folder = ['S1','S4','S5','S6','S7','S8','S9']
    model_path =  sys.argv[2]
    out_name = sys.argv[3]

    #empty lists to store the exoerimental images and some parameters, like the set and the integration radius
    x_train_list = []
    y_train_list = []
    sets = []
    int_radii = []

    for current_folder in input_sub_folder:
        input_folder = input_base + current_folder + '-20180717/' + current_folder + '/'
        #input_folder = input_base + '071618/' + current_folder + '/'
        input_images = [image for image in os.listdir(input_folder) if 'mat' in image]

        for image in input_images:
            if 'PACBED' in image:
                    #load images and haadf depth predictions, get labels
                    img_array = sio.loadmat(input_folder + image)
                    label_array = sio.loadmat(input_folder + 'HAADFprediction_r8.mat')
                    cur_radii = image.rsplit('_',1)[-1][0:2]
                    cur_set = current_folder

                    #prepare data to be looped through
                    im_fields = sio.whosmat(input_folder + image)
                    img_array = img_array[im_fields[0][0]]
                    img_array = img_array.astype('double')
                    im_range = img_array.shape[2]

                    label_fields = sio.whosmat(input_folder+'HAADFprediction_r8.mat')
                    label_array = np.squeeze(label_array[label_fields[0][0]])

                    for i in range(im_range):
                        #prepare individual array to be formatted for prediction
                        img = img_array[:,:,i]
                        img = sm.imresize(np.squeeze(img),(157, 157))
                        img = img.astype('double')
                        img = scale_range(img, 0, 1)

                        #add image and labels to prediction arrays
                        img_size = img.shape[0]
                        new_channel = np.zeros((img_size, img_size))
                        img_stack = np.dstack((img, new_channel, new_channel))
                        label = label_array[i]

                        x_train_list.append(img_stack)
                        y_train_list.append(label)
                        sets.append(cur_set)
                        int_radii.append(cur_radii)

    #load and compile the final model 

    # compile setting:
    model = load_model(model_path + 'FinalModel.h5')
    

    """ #this section is used for running experimental images through a data generator before making predictions (ie, to perform featurewise centering)
    datagen = ImageDataGenerator(featurewise_center=True)
    datagen.fit(np.asarray(x_train_list))
    generator = datagen.flow(np.asarray(x_train_list),batch_size=32)
    p_classes_2 = model.predict_generator(generator)
    """

    #grab predicted classes (maximum probability), probability distributions from model, write output to matlab file
    p_classes = model.predict_classes(np.asarray(x_train_list), batch_size=32) 
    p_arrays = model.predict(np.asarray(x_train_list), batch_size=32)
    y_list = np.asarray(y_train_list)
    p_class_list = np.asarray(p_classes)

    #print(p_class_list)
    #print(np.sum(p_arrays,axis=1))
    
    sio.savemat(out_name,{'measured':y_list,'predicted':p_class_list,'probabilities':p_arrays,sets':sets,'radii':int_radii})

def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input    

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[1])
    main()

