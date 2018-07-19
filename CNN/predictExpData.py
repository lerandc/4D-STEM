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

    input_base = '/srv/home/lerandc/Experimental Sr PACBED/' #S8-20180717/S8/'
    input_sub_folder = ['S1','S4','S5','S6','S7','S8','S9']
    model_path =  sys.argv[2]#'/srv/home/lerandc/CNN/models/071718_combined_noise/noise100_with_noiseless/'
    out_name = sys.argv[3]#'071718_noise100_with_noiseless_predictions.mat'
    x_train_list = []
    y_train_list = []
    sets = []
    int_radii = []

    for current_folder in input_sub_folder:
        input_folder = input_base + current_folder + '-20180717/' + current_folder + '/'
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
    """
        lr = 0.005
        decay = 1e-6
        momentum = 0.9
        optimizer = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
        loss = 'categorical_crossentropy'
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    """

    p_classes = model.predict_classes(np.asarray(x_train_list), batch_size=32)
    p_arrays = model.predict(np.asarray(x_train_list), batch_size=32)
    y_list = np.asarray(y_train_list)
    p_class_list = np.asarray(p_classes)
    print(p_class_list)
    print(np.sum(p_arrays,axis=1))
    sio.savemat(out_name,{'measured':y_list,'predicted':p_class_list,'probabilities':p_arrays,'sets':sets,'radii':int_radii})

def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input    

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[1])
    main()


