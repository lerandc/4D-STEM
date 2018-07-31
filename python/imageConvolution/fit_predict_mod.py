#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Splits data into test and training sets, fits models, and uses them to predict.

This file loads the data, crops images appropriately, splits the data into
appropriate test and training sets according to the run cross validation
parameters, fits models to the training sets, and uses models to predict images
from images in the convolution test set.

@author: Aidan Combs, modified by Luis Rangel DaCosta
@version: 3.0
"""
from numpy import sqrt
import math
import numpy as np
import random
from scipy.ndimage.filters import gaussian_filter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import linear_model
import statsmodels.formula.api as smf
import pandas as pd
import patsy
from timeit import default_timer as timer
from scipy.io import loadmat
from PIL import Image


def load_data(datanum, crop_pixels):
    """
    Loads appropriate dataset and crops it by a specified number of pixels.

    Parameters
    ----------
    datanum : integer indicating which dataset to load
    crop_pixels : how many rows of pixels to crop from each side of each image

    Returns
    -------
    ms_data : cropped multislice dataset
    conv_data : cropped convolution dataset
    """
    if datanum in [1, 2]:
        if datanum == 1:
            datafile1 = np.load('Cu_111_multislice.npy')
            datafile2 = np.load('Cu_111_convolution_crop.npy')

        datafile1 = datafile1[:,:,np.newaxis]
        datafile2 = datafile2[:,:,np.newaxis]
        shape = datafile1.shape
        
        ms_data = []
        conv_data = []
        for s in range(shape[2]):
            d1 = datafile1[:, :, s]
            d2 = datafile2[:, :, s]

            imageshape = d1.shape
            for i in range(imageshape[0]):
                for j in range(imageshape[1]):
                    for k in range(imageshape[2]):
                        if d2[i, j, k] < 1e-8:
                            d2[i, j, k] = 1e-8

            ms = crop(d1, crop_pixels)
            conv = crop(d2, crop_pixels)
            ms_data.append(ms)
            conv_data.append(conv)

    return ms_data, conv_data


def crop(image, pixels):
    """
    Crops an image by a certain number of pixels on each side

    Parameters
    ----------
    image : image to crop
    pixels : number of pixels to crop it by

    Returns
    -------
    new_image : cropped image
    """
    shape = image.shape
    new_image = image[pixels:shape[0]-pixels+1, pixels:shape[1]-pixels+1]
    return new_image


def blur(convdata, sigma=4, truncate=2, norm=True):
    """
    Applies a Gaussian filter with specified parameters to an image. Not used
    in data presented.

    Parameters
    ----------
    conv_data : dataset to apply filter to
    sigma : sigma value of Gaussian, default 4
    truncate : number of sigma values after which to end Gaussian, default 2
    norm : boolean indicating whether or not to conserve total intensity of
           the dataset, default True

    Returns
    -------
    blurreddata : dataset with filter applied
    """
    numofslices = len(convdata)
    blurreddata = []
    total = 0
    for s in range(numofslices):
        image = convdata[s]
        total = total + np.sum(image)
        blur = gaussian_filter(image, sigma, truncate=truncate)
        blurreddata.append(blur)

    # forces blur to conserve intensity--normalizes it to the sum of the
    # original array
    if norm is True:
        blurreddata = normalize(blurreddata, total)

    return blurreddata


def normalize(array, val):
    """
    Scale an array so its total equals a specified value. Used by blur.

    Parameters
    ----------
    array : array to normalize

    Returns
    -------
    val : value to normalize array to
    """
    total = 0
    numofslices = len(array)
    for s in range(numofslices):
        total = total + np.sum(array[s])

    normed = []
    for s in range(numofslices):
        normedslice = array[s] * (val/total)
        normed.append(normedslice)

    return normed


def build_train(thisrun):
    """
    Creates and trains a set of models according to the run parameters
    (modeltype and crossvaltype).

    Parameters
    ----------
    thisrun : RunSpecs object containing run parameters

    Returns
    -------
    model_list : list of models trained
    """
    crossvaltype = thisrun.cvtype
    modeltype = thisrun.modeltype

    if crossvaltype == 1:
        inorder = True
    else:
        inorder = False

    if crossvaltype == 1 or crossvaltype == 5:
        imagekfold(thisrun, inorder)
    elif crossvaltype == 4:
        gridblockoutkfold(thisrun)
    elif crossvaltype == 3:
        randompixelkfold(thisrun)
    elif crossvaltype == 6:
        image_leave_X_out(thisrun)

    conv_train_list = thisrun.conv_train_list
    ms_train_list = thisrun.ms_train_list

    i = 0
    model_list = []
    for conv_train in conv_train_list:
        ms_train = ms_train_list[i]
        if modeltype in [1, 2, 6, 7]:
            # Builds a polynomial model of specified degree without
            # crossterms and log term
            model, fit_time = poly(conv_train, ms_train, thisrun)
        if modeltype == 3:
            # Builds a RR model
            model, fit_time = rr(conv_train, ms_train)
        if modeltype == 4:
            # Builds a KRR model
            model, fit_time = krr(conv_train, ms_train)
        if modeltype == 5:
            # Builds a gaussian process regression model
            model, fit_time = gpr(conv_train, ms_train)
        model_list.append(model)
        i = i + 1

    thisrun.add_fit_time(fit_time)
    return model_list


def image_leave_X_out(thisrun):
    """
    Performs slicewise leave-out-X cross validation according to run
    parameters. Stores results in RunSpecs object for the run.

    Parameters
    ----------
    thisrun : RunSpecs object containing model parameters

    Returns
    -------
    None
    """
    conv_data = thisrun.conv_data
    ms_data = thisrun.ms_data
    inputs = thisrun.inputs
    trimrows = thisrun.trimrows
    rep = thisrun.loxrep
    num_of_images = len(conv_data)
    num_out = round((1-thisrun.fracin) * num_of_images)
    conv_train_list, conv_test_list = [], []
    ms_train_list, ms_test_list = [], []
    test_centers_list, train_centers_list = [], []
    out_num_list, im_in_f_list = [], []

    for i in range(rep):
        test_ims = []
        images = list(range(num_of_images))

        for j in range(num_out):
            index = random.choice(images)
            images.remove(index)
            test_ims.append(index)

        elements = 0
        totalelements = 0
        for s in range(num_of_images):
            shape = conv_data[s].shape
            elements = (elements + (shape[0]-trimrows*2) *
                        (shape[1]-trimrows*2))
            totalelements = totalelements + shape[0]*shape[1]

        num_el_test = 0
        for s in test_ims:
            shape = conv_data[s].shape
            num_el_test = (num_el_test + (shape[0] - trimrows * 2) *
                           (shape[1] - trimrows * 2))

        num_el_train = elements - num_el_test

        # Builds vector of all slices excluding specified slices
        conv_test = np.zeros([num_el_test, inputs])
        conv_train = np.zeros([num_el_train, inputs])
        ms_test = np.zeros([num_el_test, 1])
        ms_train = np.zeros([num_el_train, 1])

        traincount = 0
        test_count = 0
        for k in range(0, num_of_images):
            for i in range(trimrows, conv_data[k].shape[0]-trimrows):
                for j in range(trimrows, conv_data[k].shape[1]-trimrows):
                    if (k in test_ims) is True:
                        conv_test[test_count, :] = get_region_data(
                                inputs, conv_data, i, j, k)
                        ms_test[test_count, 0] = ms_data[k][i, j]
                        test_count = test_count + 1
                    else:
                        conv_train[traincount, :] = get_region_data(
                                inputs, conv_data, i, j, k)
                        ms_train[traincount, 0] = ms_data[k][i, j]
                        traincount = traincount + 1

        if len(test_ims) == 0:
            conv_test = conv_train
            ms_test = ms_train

        traincenters = len(ms_train)
        testcenters = len(ms_test)
        outnum = totalelements - traincenters - testcenters

        conv_test_list.append(conv_test)
        conv_train_list.append(conv_train)
        ms_test_list.append(ms_test)
        ms_train_list.append(ms_train)
        train_centers_list.append(traincenters)
        test_centers_list.append(testcenters)
        out_num_list.append(outnum)
        im_in_f_list.append(test_ims)

    thisrun.set_train_test_out_nums(np.mean(train_centers_list),
                                    np.mean(test_centers_list),
                                    np.mean(out_num_list))
    thisrun.conv_train_list = conv_train_list
    thisrun.ms_train_list = ms_train_list
    thisrun.conv_test_list = conv_test_list
    thisrun.ms_test_list = ms_test_list
    thisrun.im_in_f_list = im_in_f_list

    return


def imagekfold(thisrun, inorder):
    """
    Performs image-wise k-fold cross validation according to the run
    parameters. Stores the results in the RunSpecs object for the run.

    Parameters
    ----------
    thisrun : RunSpecs object containing parameters for the run
    inorder : boolean indicating whether the slices should be placed in
              test and training sets randomly or in order. Always False for the
              results presented (indicating random assignment).

    Returns
    -------
    None
    """
    conv_data = thisrun.conv_data
    ms_data = thisrun.ms_data
    inputs = thisrun.inputs
    trimrows = thisrun.trimrows
    num_of_images = len(conv_data)
    test_im_num = round((1-thisrun.fracin)*num_of_images)
    images, images_copy = [], []
    conv_train_list, conv_test_list = [], []
    ms_train_list, ms_test_list = [], []
    test_centers_list, train_centers_list = [], []
    out_num_list, im_in_f_list = [], []
    small_train = (thisrun.fracin < .5)

    for i in range(0, num_of_images):
        images.append(i)
        images_copy.append(i)

    if small_train:
        small_set_num = num_of_images - test_im_num
    else:
        small_set_num = test_im_num

    if test_im_num == 0:
        repeatnum = 1
    else:
        repeatnum = math.ceil(num_of_images/small_set_num)

    for i in range(0, repeatnum):
        small_set = []
        test_ims = []

        # chooses images to remove randomly from a list of slices that haven't
        # been used yet. If all have been used, repeats are allowed, but the
        # same image can't be used twice in the same fold
        if inorder is False:
            for j in range(0, small_set_num):
                if not images:
                    images = images_copy

                index = random.choice(images)
                while index in small_set:
                    index = random.choice(images)

                images.remove(index)
                small_set.append(index)

        # uses list of slices in order instead
        elif inorder is True:
            for j in range(0, small_set_num):
                if not images:
                    images = images_copy
                index = images[0]
                images.remove(index)
                small_set.append(index)

        elements = 0
        totalelements = 0
        for s in range(num_of_images):
            shape = conv_data[s].shape
            elements = elements + (shape[0]-trimrows*2) * (shape[1]-trimrows*2)
            totalelements = totalelements + shape[0]*shape[1]

        num_el_small_set = 0
        for s in small_set:
            shape = conv_data[s].shape
            num_el_small_set = (num_el_small_set + (shape[0] - trimrows * 2) *
                                (shape[1] - trimrows * 2))

        num_el_large_set = elements - num_el_small_set
        # Builds vector of all slices excluding specified slices
        if small_train:
            conv_test = np.zeros([num_el_large_set, inputs])
            conv_train = np.zeros([num_el_small_set, inputs])
            ms_test = np.zeros([num_el_large_set, 1])
            ms_train = np.zeros([num_el_small_set, 1])
        else:
            conv_train = np.zeros([num_el_large_set, inputs])
            conv_test = np.zeros([num_el_small_set, inputs])
            ms_train = np.zeros([num_el_large_set, 1])
            ms_test = np.zeros([num_el_small_set, 1])

        traincount = 0
        im_count = 0

        for k in range(0, num_of_images):
            if small_train:
                if (k in small_set) is False:
                    test_ims.append(k)
                for i in range(trimrows, conv_data[k].shape[0]-trimrows):
                    for j in range(trimrows, conv_data[k].shape[1]-trimrows):
                        if k in small_set:
                            conv_train[traincount, :] = get_region_data(
                                    inputs, conv_data, i, j, k)
                            ms_train[traincount, 0] = ms_data[k][i, j]
                            traincount = traincount + 1

                        if (k in small_set) is False:
                            conv_test[im_count, :] = get_region_data(
                                    inputs, conv_data, i, j, k)
                            ms_test[im_count, 0] = ms_data[k][i, j]
                            im_count = im_count + 1
            else:
                if (k in small_set) is True:
                    test_ims.append(k)
                for i in range(trimrows, conv_data[k].shape[0]-trimrows):
                    for j in range(trimrows, conv_data[k].shape[1]-trimrows):
                        if (k in small_set) is False:
                            conv_train[traincount, :] = get_region_data(
                                    inputs, conv_data, i, j, k)
                            print(np.asarray(ms_data).shape)
                            ms_train[traincount, 0] = ms_data[k][i, j, k]
                            traincount = traincount + 1

                        if (k in small_set) is True:
                            conv_test[im_count, :] = get_region_data(
                                    inputs, conv_data, i, j, k)
                            ms_test[im_count, 0] = ms_data[k][i, j]
                            im_count = im_count + 1

        if len(small_set) == 0:
            conv_test = conv_train
            ms_test = ms_train

        traincenters = len(ms_train)
        testcenters = len(ms_test)
        outnum = totalelements - traincenters - testcenters

        conv_test_list.append(conv_test)
        conv_train_list.append(conv_train)
        ms_test_list.append(ms_test)
        ms_train_list.append(ms_train)
        train_centers_list.append(traincenters)
        test_centers_list.append(testcenters)
        out_num_list.append(outnum)
        im_in_f_list.append(test_ims)

    thisrun.set_train_test_out_nums(np.mean(train_centers_list),
                                    np.mean(test_centers_list),
                                    np.mean(out_num_list))
    thisrun.conv_train_list = conv_train_list
    thisrun.ms_train_list = ms_train_list
    thisrun.conv_test_list = conv_test_list
    thisrun.ms_test_list = ms_test_list
    thisrun.im_in_f_list = im_in_f_list

    return


def randompixelkfold(thisrun):
    """
    Performs a random pixel k-fold cross validation according to the run
    parameters. Stores the result in the RunSpecs object for the run.

    Parameters
    ----------
    thisrun : RunSpecs object containing run parameters.

    Returns
    -------
    None
    """
    fracin = thisrun.fracin
    inputs = thisrun.inputs
    conv_data = thisrun.conv_data
    ms_data = thisrun.ms_data
    trimrows = thisrun.trimrows
    num_of_images = len(conv_data)
    total_elements = 0
    no_edge_els = 0
    small_train = (fracin < .5)

    for s in range(num_of_images):
        shape = conv_data[s].shape
        total_elements = total_elements + shape[0]*shape[1]
        no_edge_els = no_edge_els + (shape[0]-2*trimrows)*(shape[1]-2*trimrows)

    if small_train:
        num_small_set = int(fracin*no_edge_els)
        num_large_set = no_edge_els-num_small_set
        num_folds = int(math.ceil(no_edge_els/num_small_set))
    else:
        num_large_set = int(fracin*no_edge_els)
        num_small_set = no_edge_els-num_large_set
        num_folds = int(math.ceil(no_edge_els/num_small_set))

    conv_test_list, conv_train_list = [], []
    ms_test_list, ms_train_list = [], []
    test_centers_list, train_centers_list, out_num_list = [], [], []
    rands, rands_copy, rands_in_fold = [], [], []
    folds = 0

    for i in range(no_edge_els):
        rands.append(i)
        rands_copy.append(i)

    # places pixels
    for n in range(0, num_folds):
        rands_in_fold.append([])
        out = []
        for s in range(len(conv_data)):
            shape = conv_data[s].shape
            out_image = np.zeros([shape[0], shape[1]])
            out.append(out_image)
        folds = folds + 1

        for m in range(0, num_small_set):
            if not rands:
                rands = rands_copy
            # generates list of random indices that haven't already been used
            rand_index = random.choice(rands)
            while rand_index in rands_in_fold[folds-1]:
                rand_index = random.choice(rands)
            rands_in_fold[folds-1].append(rand_index)
            rands.remove(rand_index)
            randloc = pixel_index_to_loc(rand_index, trimrows, conv_data)
            # pulls out data for the current random pixel in the list. Marks
            # these locations in the out array with 2. Also marks invalid zones
            # in out array with 1. Valid squares are marked with 0.
            out[randloc[2]][randloc[0], randloc[1]] = 2

        invalid_array, invalid_count = find_invalid_no_buffer(out, trimrows)
        valid_count = total_elements - invalid_count
        # now assign training and test data
        if small_train:
            conv_test = np.zeros([num_large_set, inputs])
            ms_test = np.zeros([num_large_set, 1])
            conv_train = np.zeros([invalid_count, inputs])
            ms_train = np.zeros([invalid_count, 1])
        else:
            conv_test = np.zeros([num_small_set, inputs])
            ms_test = np.zeros([num_small_set, 1])
            conv_train = np.zeros([valid_count, inputs])
            ms_train = np.zeros([valid_count, 1])

        test_count = 0
        train_count = 0
        for k in range(0, num_of_images):
            currslice = invalid_array[k]
            for i in range(0, conv_data[k].shape[0]):
                for j in range(0, conv_data[k].shape[1]):
                    if small_train:
                        if currslice[i, j] == 0:
                            conv_test[test_count, :] = get_region_data(
                                    inputs, conv_data, i, j, k)
                            ms_test[test_count, :] = ms_data[k][i, j]
                            test_count = test_count + 1
                        elif currslice[i, j] == 2:
                            conv_train[train_count, :] = get_region_data(
                                    inputs, conv_data, i, j, k)
                            ms_train[train_count, :] = ms_data[k][i, j]
                            train_count = train_count + 1
                    else:
                        if currslice[i, j] == 2:
                            conv_test[test_count, :] = get_region_data(
                                    inputs, conv_data, i, j, k)
                            ms_test[test_count, :] = ms_data[k][i, j]
                            test_count = test_count + 1
                        elif currslice[i, j] == 0:
                            conv_train[train_count, :] = get_region_data(
                                    inputs, conv_data, i, j, k)
                            ms_train[train_count, :] = ms_data[k][i, j]
                            train_count = train_count + 1

        train_centers = len(ms_train)
        test_centers = len(ms_test)
        outnum = total_elements - no_edge_els

        conv_test_list.append(conv_test)
        conv_train_list.append(conv_train)
        ms_test_list.append(ms_test)
        ms_train_list.append(ms_train)

        train_centers_list.append(train_centers)
        test_centers_list.append(test_centers)
        out_num_list.append(outnum)

    thisrun.set_train_test_out_nums(np.mean(train_centers_list),
                                    np.mean(test_centers_list),
                                    np.mean(out_num_list))
    thisrun.conv_train_list = conv_train_list
    thisrun.ms_train_list = ms_train_list
    thisrun.conv_test_list = conv_test_list
    thisrun.ms_test_list = ms_test_list

    return


def gridblockoutkfold(thisrun):
    """
    Performs block-wise k-fold cross validation according to the run
    parameters. Stores the result in the RunSpecs object for the run.

    Places blocks of specified size randomly in a set grid calculated using the
    block size. Leaves a buffer around each block such that no point is shared
    between the input pixel grids of the test and training sets.

    Parameters
    ----------
    thisrun : RunSpecs object containing run parameters.

    Returns
    -------
    None
    """
    conv_data = thisrun.conv_data
    ms_data = thisrun.ms_data
    trimrows = thisrun.trimrows
    blocksize = thisrun.blocksize
    fracin = thisrun.fracin
    inputs = thisrun.inputs
    small_train = (fracin < .5)

    num_of_images = len(conv_data)
    buffer = trimrows*2
    grid_rows = np.zeros([num_of_images])
    grid_cols = np.zeros([num_of_images])
    trim_rows_up = np.zeros([num_of_images])
    trim_cols_left = np.zeros([num_of_images])
    total_elements = 0
    no_edge_els = 0

    for s in range(num_of_images):
        shape = conv_data[s].shape
        total_elements = total_elements + shape[0]*shape[1]
        no_edge_els = no_edge_els + (shape[0]-buffer)*(shape[1]-buffer)
        grid_rows[s] = math.floor((shape[0]-buffer)/blocksize)
        trim_rows_up[s] = math.floor(((shape[0] - buffer) % blocksize) / 2)
        grid_cols[s] = math.floor((shape[1]-buffer)/blocksize)
        trim_cols_left[s] = math.floor(((shape[1] - buffer) % blocksize) / 2)

    num_leave_in = int(fracin * no_edge_els)
    total_blocks = 0
    out_blank = []
    for s in range(num_of_images):
        total_blocks = total_blocks + int(grid_rows[s] * grid_cols[s])
        out_blank.append(np.zeros([conv_data[s].shape[0],
                                   conv_data[s].shape[1]]))

    conv_test_list = []
    conv_train_list = []
    ms_test_list = []
    ms_train_list = []
    test_centers_list = []
    train_centers_list = []
    out_num_list = []

    # blocksout = np.zeros([gridrows, gridcols, shape[2]])
    kfold_done = False
    # creates a list of indices of the blocks
    rands, rands_copy = [], []

    for i in range(0, total_blocks):
        rands.append(i)
        rands_copy.append(i)

    # if either gridrows or gridcols is 0, then the block size is too big.
    if 0 in grid_rows or 0 in grid_cols:
        kfold_done = True

    folds = 0
    rands_in_fold = []

    while kfold_done is False:
        folds = folds + 1
        num_blocks = 0
        rands_in_fold.append([])
        out = []
        for s in range(num_of_images):
            out.append(np.zeros([conv_data[s].shape[0],
                                 conv_data[s].shape[1]]))

        done = False
        # places blocks
        while done is False:
            # increments number of blocks by one
            num_blocks = num_blocks + 1
            # generates a random index that hasn't already been used
            if len(rands) == 0:
                rands = rands_copy
                kfold_done = True

            rand_index = random.choice(rands)
            while rand_index in rands_in_fold[folds-1]:
                rand_index = random.choice(rands)
            rands_in_fold[folds-1].append(rand_index)
            rands.remove(rand_index)
            rand_block_loc = block_index_to_loc(rand_index, grid_rows,
                                                grid_cols)
            rand_loc = block_loc_to_grid_loc(rand_block_loc[0],
                                             rand_block_loc[1],
                                             rand_block_loc[2], blocksize,
                                             buffer, trim_rows_up,
                                             trim_cols_left)

            # pulls out data for the current random block in the list. Marks
            # these locations in the out array with 2. Also marks invalid
            # zones in out array with 1. Valid squares are marked with 0.
            for ii in range(0, blocksize):
                for jj in range(0, blocksize):
                    k = rand_loc[2]
                    out_image = out[k]
                    out_image[rand_loc[0]+ii, rand_loc[1]+jj] = 2
                    out[k] = out_image
            # determines elements that are invalid and whether to continue
            # loop. Stops loop when enough elements have been taken out or
            # when there are no more valid blocks remaining.
            invalid_array, invalid_count = find_invalid(out, buffer)
            valid_count = total_elements-invalid_count
            if num_leave_in >= valid_count:
                done = True

        # now assign training and test data
        small_set_num = num_blocks * blocksize * blocksize

        if small_train:
            conv_test = np.zeros([valid_count, inputs])
            ms_test = np.zeros([valid_count, 1])
            conv_train = np.zeros([small_set_num, inputs])
            ms_train = np.zeros([small_set_num, 1])
        else:
            conv_test = np.zeros([small_set_num, inputs])
            ms_test = np.zeros([small_set_num, 1])
            conv_train = np.zeros([valid_count, inputs])
            ms_train = np.zeros([valid_count, 1])

        # assigns large set data by checking out matrix for squares marked
        # valid
        large_set_count = 0
        for k in range(0, num_of_images):
            curr_slice = invalid_array[k]
            shape = curr_slice.shape
            for i in range(0, shape[0]):
                for j in range(0, shape[1]):
                    if curr_slice[i, j] == 0:
                        if small_train:
                            conv_test[large_set_count, :] = get_region_data(
                                inputs, conv_data, i, j, k)
                            ms_test[large_set_count, :] = ms_data[k][i, j]
                        else:
                            conv_train[large_set_count, :] = get_region_data(
                                inputs, conv_data, i, j, k)
                            ms_train[large_set_count, :] = ms_data[k][i, j]
                        large_set_count = large_set_count + 1

        # for each block removed, add its data to the small set. Allows for
        # repeated blocks, which just searching the out matrix for squares
        # marked two would not do.
        small_set_count = 0
        for n in rands_in_fold[folds-1]:
            rand_block_loc = block_index_to_loc(n, grid_rows, grid_cols)
            rand_loc = block_loc_to_grid_loc(
                    rand_block_loc[0], rand_block_loc[1], rand_block_loc[2],
                    blocksize, buffer, trim_rows_up, trim_cols_left)
            for ii in range(0, blocksize):
                for jj in range(0, blocksize):
                    i = rand_loc[0] + ii
                    j = rand_loc[1] + jj
                    k = rand_loc[2]
                    if small_train:
                        conv_train[small_set_count, :] = get_region_data(
                                inputs, conv_data, i, j, k)
                        ms_train[small_set_count, :] = ms_data[k][i, j]
                    else:
                        conv_test[small_set_count, :] = get_region_data(
                                inputs, conv_data, i, j, k)
                        ms_test[small_set_count, :] = ms_data[k][i, j]
                    small_set_count = small_set_count + 1

        # determines whether all data has been part of test set and
        # therefore whether to continue loop
        if not rands:
            kfold_done = True
        train_centers = len(ms_train)
        test_centers = len(ms_test)
        out_num = total_elements - train_centers - test_centers

        conv_test_list.append(conv_test)
        conv_train_list.append(conv_train)
        ms_test_list.append(ms_test)
        ms_train_list.append(ms_train)

        train_centers_list.append(train_centers)
        test_centers_list.append(test_centers)
        out_num_list.append(out_num)

    if 0 in grid_cols or 0 in grid_rows:
        train_centers = 0
        test_centers = 0
        out_num = 0
    else:
        train_centers = np.mean(train_centers_list)
        test_centers = np.mean(test_centers_list)
        out_num = np.mean(out_num_list)

    thisrun.set_train_test_out_nums(train_centers, test_centers, out_num)
    thisrun.conv_train_list = conv_train_list
    thisrun.ms_train_list = ms_train_list
    thisrun.conv_test_list = conv_test_list
    thisrun.ms_test_list = ms_test_list

    return


def get_region_data(inputs, data, i, j, k):
    """
    Given the dataset and index of a point, gets that point and a specified
    number of its nearest neighbors.

    Parameters
    ----------
    inputs : number of points to return
    data : dataset
    i : row index
    j : column index
    k : image index

    Returns
    -------
    datalist : list of points in the area of the specified point
    """
    index = math.floor(sqrt(inputs)/2)
    datalist = np.zeros(inputs)
    count = 0
    for ii in range(i-index, i+index+1):
        for jj in range(j-index, j+index+1):
            dataslice = data[k]
            datalist[count] = dataslice[ii, jj, k]
            count = count + 1
    return datalist


def find_invalid_no_buffer(out, trimrows):
    """
    Finds points in the dataset which cannot be centers due to being too close
    to the edge or having been selected for either the test or training set
    already. Does not leave any buffer around points. Used for random pixel
    cross validation.

    Parameters
    ----------
    out : array of the same size as the dataset with points that have already
          been placed into the test or training set marked.
    trimrows : number of rows at the edge of the image which cannot be centers
               because they can't have the appropriately sized input pixel
               grid. Equal to (inp_side_len - 1)/ 2.

    Returns
    -------
    The edited out array and a count of the invalid points.
    """
    numofslices = len(out)
    trimrows = int(trimrows)
    totalelements = 0
    for k in range(0, numofslices):
        currslice = out[k]
        shape = currslice.shape
        totalelements = totalelements + shape[0]*shape[1]
        for i in range(0, trimrows):
            for j in range(0, shape[1]):
                if currslice[i][j] == 0:
                    currslice[i][j] = 1
                if currslice[shape[0]-1-i][j] == 0:
                    currslice[shape[0]-1-i][j] = 1
        for j in range(0, trimrows):
            for i in range(0, shape[0]):
                if currslice[i][j] == 0:
                    currslice[i][j] = 1
                if currslice[i][shape[1]-j-1] == 0:
                    currslice[i][shape[1]-j-1] = 1
        out[k] = currslice

    validcount = 0
    for k in range(0, numofslices):
        currslice = out[k]
        shape = currslice.shape
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                if currslice[i][j] == 0:
                    validcount = validcount + 1

    invalidcount = totalelements - validcount
    return out, invalidcount


def find_invalid(out, buffer):
    """
    Finds points in the dataset which cannot be centers due to being too close
    to the edge, having been selected for the test or training set already, or
    being in a previously set buffer zone. Used for random block cross
    validation.

    Parameters
    ----------
    out : array the same size as the dataset with pixels already in the test
          or training set marked
    buffer : number of pixels that need to be left as a buffer around
             blocks. Equal to inp_side_len - 1.

    Returns
    -------
    invalidarray : array with invalid pixels marked
    invalidcount : number of invalid pixels.
    """
    numofslices = len(out)
    invalidarray = []
    for s in range(numofslices):
        currshape = out[s].shape
        invalidarray.append(np.zeros([currshape[0], currshape[1]]))

    invalidcount = 0
    edgedist = int(buffer/2)

    for k in range(0, numofslices):
        shape = out[k].shape
        outslice = out[k]
        invalidslice = invalidarray[k]
        # fills in squares that are invalid because of proximity to edge
        for i in range(0, shape[0]):
            for j in range(shape[1] - edgedist, shape[1]):
                if invalidslice[i, j] == 0:
                    invalidslice[i, j] = 1
                    invalidcount = invalidcount + 1
            for j in range(0, edgedist):
                if invalidslice[i, j] == 0:
                    invalidslice[i, j] = 1
                    invalidcount = invalidcount + 1

        for j in range(0, shape[1]):
            for i in range(0, edgedist):
                if invalidslice[i, j] == 0:
                    invalidslice[i, j] = 1
                    invalidcount = invalidcount + 1
            for i in range(shape[0]-edgedist, shape[0]):
                if invalidslice[i, j] == 0:
                    invalidslice[i, j] = 1
                    invalidcount = invalidcount + 1

        # fills in squares that are invalid because of proximity to block
        for j in range(edgedist, shape[1] - edgedist):
            for i in range(edgedist, shape[0] - edgedist):
                if outslice[i, j] == 2:
                    # marks the centerpoint with a 2. These points are
                    # marked as invalid.
                    if invalidslice[i, j] == 0:
                        invalidcount = invalidcount + 1
                    invalidslice[i, j] = 2

                    if buffer != 0:
                        starti = i-buffer
                        startj = j-buffer
                        endi = i+buffer+1
                        endj = j+buffer+1
                        if starti < 0:
                            starti = 0
                        if startj < 0:
                            startj = 0
                        if endi >= shape[0]:
                            endi = shape[0]-1
                        if endj >= shape[1]:
                            endj = shape[1]-1

                        # finds surrounding points, marks them with a 1
                        for ii in range(starti, endi):
                            for jj in range(startj, endj):
                                if invalidslice[ii, jj] == 0:
                                    invalidslice[ii, jj] = 1
                                    invalidcount = invalidcount + 1
        invalidarray[k] = invalidslice

    return invalidarray, invalidcount


def pixel_index_to_loc(index, trimrows, convdata):
    """
    Takes an integer indicating the index of a pixel and converts it into a
    tuple giving the pixel's location.

    Parameters
    ----------
    index : integer representing pixel index
    trimrows : how many pixels to leave around the edge of the image
    convdata : convolution dataset

    Returns
    -------
    [i, j, k] : tuple representing location of specified point in dataset
    """
    foundslice = False
    totalel = 0
    k = 0
    while foundslice is False:
        currslice = convdata[k]
        elinslice = ((currslice.shape[0] - trimrows * 2) *
                     (currslice.shape[1] - trimrows * 2))
        totalel = totalel + elinslice
        if totalel > index:
            foundslice = True
        else:
            k = k+1
    shape = convdata[k].shape
    index = int(index - (totalel-elinslice))
    i = int(math.floor(index/(shape[1]-2*trimrows))) + trimrows
    index = index % (shape[1] - 2*trimrows)
    j = int(index) + trimrows
    return[i, j, k]


def block_index_to_loc(index, gridrows, gridcols):
    """
    Takes an integer representing the index of a block and returns a tuple
    representing its location in the block grid. Used in random block cross
    validation.

    Parameters
    ----------
    index : integer representing the index of the block
    gridrows : number of rows in the block grid, per image
    gridcols: number of columns in the block grid, per image

    Returns
    -------
    [i, j, k] : tuple representing location of block in the block grid
    """
    foundslice = False
    totalblocks = 0
    k = 0
    while foundslice is False:
        rows = gridrows[k]
        cols = gridcols[k]
        blocksinslice = rows * cols
        totalblocks = totalblocks + blocksinslice
        if totalblocks > index:
            foundslice = True
        else:
            k = k+1
    index = int(index - (totalblocks-blocksinslice))
    i = int(math.floor(index/cols))
    index = index % cols
    j = int(index)
    return [i, j, k]


def block_loc_to_grid_loc(i, j, k, blocksize, buffer, trimrowsup,
                          trimcolsleft):
    """
    Takes a location in the block grid and finds the location of its upper left
    corner in the dataset. Used in random block cross validation.

    Parameters
    ----------
    i : row index of block
    j : column index of block
    k : image index of block
    blocksize : the side length of the block
    buffer : how many pixels need to be left around blocks in different sets
             to ensure input pixels grids do not overlap.
    trimrowsup : number of rows on top that are not part of the block grid
    trimcolsleft : number of columns on the left hand side that are not part of
                   block grid

    Returns
    -------
    [i, j, k] : tuple representing location of upper left corner of block
    """
    trimup = trimrowsup[k]
    trimleft = trimcolsleft[k]
    i = int(i*blocksize + buffer/2 + trimup)
    j = int(j*blocksize + buffer/2 + trimleft)
    return [i, j, k]


def find_term_nums(thisrun):
    """
    Finds the number of terms in the models being tested by this run by using
    the model parameters in the RunSpecs object.

    Parameters
    ----------
    thisrun : RunSpecs object containing model parameters for the run

    Returns
    -------
    terms : number of terms in the model
    """
    deg = thisrun.deg
    inputs = thisrun.inputs
    modeltype = thisrun.modeltype
    conv_train = np.ones([inputs, inputs])
    if modeltype == 1:
        crossterms = False
        logterm = False
        terms = inputs * deg + 1
    elif modeltype == 2:
        crossterms = True
        logterm = False
    elif modeltype == 6:
        crossterms = False
        logterm = True
        terms = (inputs + 1) * deg + 1
    elif modeltype == 7:
        crossterms = True
        logterm = True
    else:
        crossterms = False
        terms = 0

    if crossterms is True:
        if logterm is True:
            conv_train = add_log(conv_train)
        polyn = PolynomialFeatures(degree=deg)
        polyn.fit_transform(conv_train)
        terms = polyn.n_output_features_
    return terms


def find_term_nums_direct(deg, s, modeltype):
    """
    Finds the number of terms in a model using the degree, input grid side
    length, and modeltype directly as inputs (means that it is not necessary
    to first create a RunSpecs object).

    Parameters
    ----------
    deg : degree of the model
    s : input grid side length
    modeltype : integer indicating the model type.
                1=polynomial, no cross terms, no log term
                2=polynomial, cross terms, no log term
                6=polynomial, no cross terms, log term
                7=polynomial, cross terms, log term

    Returns
    -------
    terms : number of terms in the model
    """
    inputs = s*s
    conv_train = np.ones([inputs, inputs])
    if modeltype == 1:
        crossterms = False
        logterm = False
        terms = inputs * deg + 1
    elif modeltype == 2:
        crossterms = True
        logterm = False
    elif modeltype == 6:
        crossterms = False
        logterm = True
        terms = (inputs + 1) * deg + 1
    elif modeltype == 7:
        crossterms = True
        logterm = True
    else:
        crossterms = False
        terms = 0

    if crossterms is True:
        if logterm is True:
            conv_train = add_log(conv_train)
        polyn = PolynomialFeatures(degree=deg)
        polyn.fit_transform(conv_train)
        terms = polyn.n_output_features_
    return terms


def poly(conv_train, ms_train, thisrun):
    """
    Fits a polynomial model of specified degree, number of inputs, and type
    to the training data.

    Parameters
    ----------
    conv_train : convolution training set
    ms_train : multislice training set
    thisrun : RunSpecs object containing model parameters

    Returns
    -------
    polyfit : trained model
    time : time taken to train model (seconds)
    """
    model_type = thisrun.modeltype
    inputs = thisrun.inputs
    deg = thisrun.deg
    logterm = False
    cross = False
    if model_type in [6, 7]:
        logterm = True
    if model_type in [2, 7]:
        cross = True

    if cross is False:
        names, completenames, formula = get_formula(deg, inputs, False,
                                                    logterm)

        if logterm is True:
            conv_train = add_log(conv_train)
        df = pd.DataFrame(data=np.column_stack((conv_train, ms_train)),
                          columns=completenames)
        start = timer()
        polyfit = smf.ols(formula, df).fit()
        end = timer()
        time = end - start

    if cross is True:
        # fits model using cross terms
        if logterm is True:
            conv_train = add_log(conv_train)
        poly = PolynomialFeatures(degree=deg)
        conv_train_transform = poly.fit_transform(conv_train)

        # fits model using data from transformed vector
        polyfit = linear_model.LinearRegression()
        start = timer()
        polyfit.fit(conv_train_transform, ms_train)
        end = timer()
        time = end - start

    return polyfit, time


def rr(conv_train, ms_train):
    """
    Fits ridge regression model to training data

    Parameters
    ----------
    conv_train : convolution training data
    ms_train : multislice training data

    Returns
    -------
    rrfit : model
    time : time taken to fit model
    """
    rrfit = Ridge(alpha=1)
    start = timer()
    rrfit.fit(conv_train, ms_train)
    end = timer()
    time = end - start
    return rrfit, time


def krr(conv_train, ms_train):
    """
    Fits kernel ridge regression model to training data

    Parameters
    ----------
    conv_train : convolution training data
    ms_train : multislice training data

    Returns
    -------
    krrfit : model
    time : time taken to fit model
    """
    krrfit = KernelRidge(alpha=1)
    start = timer()
    krrfit.fit(conv_train, ms_train)
    end = timer()
    time = end - start
    return krrfit, time


def gpr(conv_train, ms_train):
    """
    Fits Gaussian process regression model to training data

    Parameters
    ----------
    conv_train : convolution training data
    ms_train : multislice training data

    Returns
    -------
    gprfit : model
    time : time taken to fit model
    """
    gprfit = GaussianProcessRegressor(kernel=None)
    start = timer()
    gprfit.fit(X=conv_train, y=ms_train)
    end = timer()
    time = end - start
    return gprfit, time


def predict(conv_test, model, thisrun):
    """
    Takes a fitted model and a test set and predicts images.

    Parameters
    ----------
    conv_test : convolution test set
    model : model to use for prediction

    Returns
    -------
    predicted : list of predicted pixel intensity values
    time : time taken to apply model (seconds per pixel)
    """
    modeltype = thisrun.modeltype
    deg = thisrun.deg
    inputs = thisrun.inputs

    if modeltype == 1 or modeltype == 6:
        if modeltype == 1:
            logterm = False
        if modeltype == 6:
            logterm = True
            conv_test = add_log(conv_test)
        names, completenames, formula = get_formula(deg, inputs, True, logterm)
        df = pd.DataFrame(data=conv_test, columns=names)
        x = patsy.dmatrix(formula, data=df)
        start = timer()
        predicted = model.predict(x, transform=False)
        end = timer()
        time = (end - start)/len(predicted)

    if modeltype is 2 or modeltype is 7:
        poly = PolynomialFeatures(degree=deg)
        if modeltype == 7:
            conv_test = add_log(conv_test)
        conv_test_transform = poly.fit_transform(conv_test)
        start = timer()
        predicted = model.predict(conv_test_transform)
        end = timer()
        time = (end - start)/len(predicted)

    elif modeltype == 3 or modeltype == 4 or modeltype == 5:
        start = timer()
        predicted = model.predict(conv_test)
        end = timer()
        time = (end - start)/len(predicted)
    thisrun.add_predicted(predicted)

    return predicted, time


def get_formula(deg, inputs, predict, logterm):
    """
    Constructs the correct formula for polynomials without cross terms given
    model parameters.

    Parameters
    ----------
    deg : degree of polynomial
    inputs : number of pixels in input pixel grid
    predict : boolean indicating whether to include 'ms ~' at beginning
    logterm : boolean indicating whether to include a log term

    Returns
    -------
    names : list of pixels to use
    completenames : list of pixels to use + ms term
    formula : String representing complete formula
    """
    names = []
    completenames = []
    center = math.floor(inputs/2)
    for n in range(inputs):
        names.append('p'+str(n))
        completenames.append('p'+str(n))
    if logterm is True:
        names.append('np.log(p' + str(center) + ')')
        completenames.append('np.log(p' + str(center) + ')')
    completenames.append('ms')

    if predict is False:
        formula = 'ms ~ '
    else:
        formula = ''

    count = 0
    for n in names:
        for d in range(deg):
            if count != 0:
                formula = formula + ' + np.power('+str(n)+', '+str(d+1)+')'
            else:
                formula = formula + 'np.power('+str(n)+', '+str(d+1)+')'
            count = count + 1

    return names, completenames, formula


def add_log(x):
    """
    Adds a column into an array of pixel intensities that represents the log
    of the intensity of the center pixel.

    Parameters
    ----------
    x : array to add intensity of the center pixel column into

    Returns
    -------
    newx : array with log column added
    """
    inputs = x.shape[1]
    center = math.floor(inputs/2)
    newx = np.zeros([x.shape[0], x.shape[1]+1])
    newx[:, :-1] = x
    newx[:, -1:] = np.reshape(np.log10(x[:, center]), [x.shape[0], 1])
    return newx
