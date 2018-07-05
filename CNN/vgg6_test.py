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
import time

def main():
    start_time = time.time()

    K.tensorflow_backend._get_available_gpus()
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    input_folder = '/srv/home/lerandc/CNN/s8_ph0_Ti_pacbed_1111_npy_noise_peak_1/'
    result_path =  '/srv/home/lerandc/CNN/s8_ph0_Ti_pacbed_1111_npy_noise_peak_1/results/'
    input_images = [image for image in os.listdir(input_folder) if 'npy' in image]

    x_train_list = []
    y_train_list = []

    sx, sy = 0, 0

    for image in input_images:
        cmp = image.split('_')
        label = int(cmp[2][:-4])
        r = int(cmp[0].split('-')[1][1:])
        if r < 8:
            img = np.load(input_folder + image)
            img_size = img.shape[0]
            sx, sy = img.shape[0], img.shape[1]
            new_channel = np.zeros((img_size, img_size))
            img_stack = np.dstack((img, new_channel, new_channel))
            x_train_list.append(img_stack)
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
    np.save(input_folder + 'results/y_train.npy', y_train)


    logs = [log for log in os.listdir(result_path) if 'log' in log]
    max_index = 0
    for log in logs:
        cur = int(log.split('_')[1])
        if cur > max_index:
            max_index = cur
    max_index = max_index + 1

    batch_size = 32
    # step 1
    save_bottleneck_features(x_train, y_train, batch_size, nb_train_samples)

    # step 2
    epochs = 12
    batch_size = 32  # batch size 32 works for the fullsize simulation library which has 19968 total files, total number of training file must be integer times of batch_size
    train_top_model(y_train, nb_class, max_index, epochs, batch_size, input_folder, result_path)

    # step 3
    epochs = 50
    batch_size = 32
    fine_tune(x_train, y_train, sx, sy, max_index, epochs, batch_size, input_folder, result_path)

    print('Total computing time is: ')
    print(int((time.time() - start_time) * 100) / 100.0)


def save_bottleneck_features(x_train, y_train, batch_size, nb_train_samples):
    model = applications.VGG16(include_top=False, weights='imagenet')
    print('before featurewise center')
    datagen = ImageDataGenerator(
        featurewise_center=True)

    datagen.fit(x_train)
    print('made it past featurewise center')
    generator = datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=False)
    print('made it past generator')
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    print('made it past the bottleneck features')
    np.save('/srv/home/lerandc/CNN/s8_ph0_Ti_pacbed_1111_npy_noise_peak_1/results/bottleneck_features_train.npy',
            bottleneck_features_train)

def train_top_model(y_train, nb_class, max_index, epochs, batch_size, input_folder, result_path):
    train_data = np.load('/srv/home/lerandc/CNN/s8_ph0_Ti_pacbed_1111_npy_noise_peak_1/results/bottleneck_features_train.npy')
    train_labels = y_train
    print(train_data.shape, train_labels.shape)
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(nb_class, activation='sigmoid'))

    # compile setting:
    lr = 0.005
    decay = 1e-6
    momentum = 0.9
    optimizer = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    loss = 'categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    # model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])

    bottleneck_log = '/srv/home/lerandc/CNN/s8_ph0_Ti_pacbed_1111_npy_noise_peak_1/results/' + 'training_' + str(max_index) + '_bnfeature_log.csv'
    csv_logger_bnfeature = callbacks.CSVLogger(bottleneck_log)
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=3, verbose=1, mode='auto')

    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2,
              callbacks=[csv_logger_bnfeature, earlystop])
    with open(bottleneck_log, 'a') as log:
        log.write('\n')
        log.write('input images: ' + input_folder + '\n')
        log.write('batch_size:' + str(batch_size) + '\n')
        log.write('learning rate: ' + str(lr) + '\n')
        log.write('learning rate decay: ' + str(decay) + '\n')
        log.write('momentum: ' + str(momentum) + '\n')
        log.write('loss: ' + loss + '\n')

    model.save_weights( result_path + 'bottleneck_fc_model.h5')

def fine_tune(x_train, y_train, sx, sy, max_index, epochs, batch_size, input_folder, result_path):
    train_data = x_train
    train_labels = y_train
    print(train_data.shape, train_labels.shape)

    model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(sx, sy, 3))
    print('Model loaded')

    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dense(52, activation='sigmoid'))

    top_model.load_weights(result_path + 'bottleneck_fc_model.h5')

    new_model = Sequential()
    for l in model.layers:
        new_model.add(l)
    new_model.add(top_model)

    # for layer in new_model.layers[:6]:
    # layer.trainable = False

    # compile settings
    lr = 0.0001
    decay = 1e-6
    momentum = 0.9
    optimizer = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    loss = 'categorical_crossentropy'
    new_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    fineture_log = '/srv/home/lerandc/CNN/s8_ph0_Ti_pacbed_1111_npy_noise_peak_1/results/' + 'training_' + str(max_index) + '_finetune_log.csv'
    csv_logger_finetune = callbacks.CSVLogger(fineture_log)
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')

    new_model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2,
                  callbacks=[csv_logger_finetune, earlystop])

    with open(fineture_log, 'a') as log:
        log.write('\n')
        log.write('input images: ' + input_folder + '\n')
        log.write('batch_size:' + str(batch_size) + '\n')
        log.write('learning rate: ' + str(lr) + '\n')
        log.write('learning rate decay: ' + str(decay) + '\n')
        log.write('momentum: ' + str(momentum) + '\n')
        log.write('loss: ' + loss + '\n')

    new_model.save('/srv/home/lerandc/CNN/s8_ph0_Ti_pacbed_1111_npy_noise_peak_1/results/FinalModel.h5')  # save the final model for future loading and prediction





# step 4 make predictions using experiment results

if __name__ == '__main__':
    main()
