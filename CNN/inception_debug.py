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

model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(sx, sy, 3))
print('Model loaded')

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
#top_model.add(Dropout(0.3))
top_model.add(Dense(52, activation='sigmoid'))

top_model.load_weights(result_path + 'bottleneck_fc_model.h5')

new_model = Sequential()
for l in model.layers:
    new_model.add(l)
#new_model.add(Dropout(0.3))
new_model.add(top_model)