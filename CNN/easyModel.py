
import os
import sys
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import callbacks
from keras import optimizers
from keras import backend as K
import tensorflow as tf
import numpy as np
import scipy.io as sio
import time

def main():
    start_time = time.time()

    input_base = '/srv/home/lerandc/outputs/712_STO/'
    input_sub_folder = ['0_0/','05_0/','025_025/','1_0/','1_1/','2_0/','2_2/','3_0/']

    noise = ['noiseless']
    input_sub_folder = [folder+'pacbeds/'+noise+'/' for folder in input_sub_folder] #adds pacbed folder to allow easy change of sub folders

    result_path =  '/srv/home/lerandc/CNN/models/'

    train_data,train_labels = getInputData(input_base=input_base,input_sub_folder=input_sub_folder)

    model = createModel(input_base,input_sub_folder)

    batch_size = 32
    epochs = 50
    lr = 0.001
    decay = lr/epochs
    optimizer = createOptimizer(lr=lr,decay=decay,momentum=0.9,nesterov=True)

    loss = 'categorical_crossentropy'
    metrics =['accuracy']
    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)

    steps_per_epoch = len(train_data)/batch_size #can be arbitrary integer, this loops through all data once
    validation_steps = steps_per_epoch//5 #effectively 0.2 validation split, // is integer divide, use validation steps only if using generator for validation
                                        
    generator = createGenerator(train_data,train_labels,batch_size)

    #establish various callbacks
    finetune_log = result_path + 'finetune_log.csv'
    csv_logger_finetune = callbacks.CSVLogger(finetune_log)
    earlystop = callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')
    
    filepath='FinalModel_best_weights.h5'
    checkpoint = callbacks.ModelCheckpoint(filepath=result_path+filepath,monitor='val_acc',verbose=0,save_best_only=True)

    lr_reduce = callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=5,min_delta=0.01,min_lr=lr/1000,verbose=1)

    history = model.fit_generator(generator,epochs=epochs,steps_per_epoch=steps_per_epoch,
                                validation_data=generator,validation_steps=validation_steps,
                                callbacks=[csv_logger_finetune, earlystop, checkpoint,lr_reduce],verbose=2)

    model.save(result_path + 'FinalModel.h5')

    with open(finetune_log, 'a') as log:
        log.write('\n')
        log.write('input images: ' + input_base + '\n')
        log.write('batch_size:' + str(batch_size) + '\n')
        log.write('learning rate: ' + str(lr) + '\n')
        log.write('learning rate decay: ' + str(decay) + '\n')
        log.write('momentum: ' + str(momentum) + '\n')
        log.write('loss: ' + loss + '\n')

    print('Total computing time is: ')
    print(int((time.time() - start_time) * 100) / 100.0)


def getInputData(input_base,input_sub_folder):
    #loads input data from folder(s) into memory
    x_train_list = []
    y_train_list = []

    sx, sy = 0, 0

    for current_folder in input_sub_folder:
        input_folder = input_base + current_folder
        input_images = [image for image in os.listdir(input_folder) if 'Sr_PACBED' in image]

        for image in input_images:
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

    return x_train, y_train

def createModel(weights='imagenet',include_top=False,input_shape=(157,157,3),nClasses=91):
    #creates model architecture and loads weights
    #I use VGG16 here with the top model based on my extra dense sereis
    base_model = applications.VGG16(weights=weights,include_top=include_top,input_shape=input_shape)

    #create top model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dropout(0.3))
    top_model.add(Dense(nClasses*4,activation='relu'))
    top_model.add(Dropout(0.3))
    top_model.add(Dense(nClasses*4,activation='relu'))
    top_model.add(Dense(nClasses,activation='softmax'))

    final_model = Sequential()
    final_model.add(base_model)
    final_model.add(top_model)

    return final_model

def createOptimizer(lr=0.001,decay=1e-6,momentum=0.9,nesterov=True):
    #I used SGD for everything, most other optimizers have less parameters to fidddle with
    #default values I pass are randomly chosen from one of scripts and not gauranteed to work well
    optimizer = optimizers.SGD(lr=lr,decay=decay,momentum=momentum,nesterov=nesterov)

    return optimizer

def createGenerator(train_data,train_labels,batch_size,seed=None,featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False,
    zca_whitening=False, zca_epsilon=1e-06, rotation_range=0.0, width_shift_range=0.0, height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0, channel_shift_range=0.0,
    fill_mode='nearest', cval=0.0, horizontal_flip=False, vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, validation_split=0.0):
    #creates data augmentation generator to be used during training
    #will supply default values to the generator unless otherwise specified

    datagen = ImageDataGenerator(
        featurewise_center=featurewise_center,
        samplewise_center=samplewise_center,
        featurewise_std_normalization=featurewise_std_normalization,
        samplewise_std_normalization=samplewise_std_normalization,
        zca_epsilon=zca_epsilon,
        zca_whitening=zca_whitening,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        channel_shift_range=channel_shift_range,
        fill_mode=fill_mode,
        cval=cval,
        horizontal_flip=horizontal_flip,
        preprocessing_function=preprocessing_function,
        data_format=data_format,
        validation_split=validation_split)        

    #augment is default false, I never played with it but maybe it could be useful
    datagen.fit(x_train,augment=None,seed=seed)
        
    #seed is default None, could be useful to control reproducibility    
    generator = datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        seed=seed)
    
    return generator

def scale_range (image, min_val, max_val):
    image += -(np.min(image))
    image /= np.max(image) / (max_val - min_val)
    image += min_val
    return image 

if __name__ == '__main__':
    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    #os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[1])
    main()
