#One time script to see if averaging noisy experimental data together would improve prediction quality
#Author: Luis Rangel DaCosta, lerandc@umich.edu
#Last comment date: 7-31-2018

import os
import sys
import numpy as np 
import scipy.io as sio

#grab all images
current_folder = 'C:/Users/leran/Desktop/Simulations and Data/7-9/Experimental Sr PACBED/Sr-PACBEDs/Sr_50pm/'
input_images = [image for image in os.listdir(current_folder) if (('npy' in image) and ('Sr' in image))]
slice = 1.9525


#get a list of the positions
images  = []
for image in input_images:
    cmp = image.split('_')
    images.append([image,cmp[-1][:-4]])

#sort by the positions
images.sort(key=lambda x: x[1],reverse=False)

#set indentifiers to correspond images arbirtrarily sized groups of images together
markers = [0]
cur_y = images[0][1]
mark = -1
for x,y in images:
    mark += 1
    if not (y == cur_y):
        markers.append(mark)
        cur_y = y

#average all images by their groups
markers.append(len(images))
for x in range(len(markers)-1):
    img_subset = images[markers[x]:markers[x+1]][:]
    tmp = np.zeros(np.load(current_folder + img_subset[0][0][:]).shape)
    for img,depth in img_subset:    
        new_img = np.load(current_folder+img)
        tmp += new_img
    tmp /= len(img_subset)
    sio.savemat(current_folder+'Sr_50pm_avg_' + images[markers[x]][1][:] +'.mat',{'pacbed':tmp})