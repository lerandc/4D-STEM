#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains methods that get output information for image simulation tests.

This file contains methods that characterize the accuracy of the model.
Broadly, some methods calculate the error of the model by comparing the
predicted pixel intensity values with the actual pixel intensity values (the
values in the multislice data test set), some methods generate plots that
display errors in various ways, and some methods are supporting methods for
one or both of these purposes.

@author: Aidan Combs
@version: 3.0
"""

from numpy import square, mean, sqrt
import numpy as np
from fractions import Fraction
from functools import reduce
from operator import mul
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
import math
from sklearn.metrics import r2_score
from decimal import Decimal


def get_crystallity_and_error_maps(im_in_f_list, thisrun):
    """
    Gets error information for each composition class and makes error maps.

    Creates error vectors (root mean square and 1-r^2) for each composition
    class (amorphous, crystalline, and mixed) using information from
    im_in_f_list to determine which error vectors correspond to which images.
    Adds these vectors to the RunSpecs object passed. Designed to work with
    image-wise k-fold cross validation.

    Also gets images for each image index in im_in_f_list and creates one error
    map figure (containing four subplots) for each by calling error_maps.

    Parameters
    ----------
        im_in_f_list : list of lists of image indices
            a list containing, for each repetition (fold), a list of the
            indices of images in that test set.
        thisrun : RunSpecs object
            contains data and model parameters for the run.

    Returns
    -------
        None"""
    amorph_rms, mixed_rms, xtal_rms = [], [], []
    amorph_rsq, mixed_rsq, xtal_rsq = [], [], []
    inputs = thisrun.inputs
    i = 0
    # Loop over all folds
    for ims_in_fold in im_in_f_list:
        # ms_test and predicted here are the target and predicted values
        # (respectively) for the fold in question.
        ms_test = thisrun.ms_test_list[i]
        predicted = thisrun.predicted_list[i]
        i += 1
        in_order_ims = []
        shapes = []
        total_ims = len(thisrun.ms_data)

        # Gets shapes of the image arrays for all images in the test set.
        for s in range(total_ims):
            if s in ims_in_fold:
                shapes.append(thisrun.ms_data[s].shape)
                in_order_ims.append(s)

        # Gets lists of data vectors (error, which is predicted minus actual,
        # actual, and predicted) in fold.
        dif_vec_list, ms_vec_list, pred_vec_list = get_im_vecs(
            predicted, ms_test, ims_in_fold, shapes, inputs)

        # Gets lists of images (error, actual, and predicted) in fold.
        dif_im_list, ms_im_list, pred_im_list = get_ims(
            predicted, ms_test, ims_in_fold, shapes, inputs)

        # Loop over each image in fold.
        for j in range(len(ims_in_fold)):
            ms_vec = ms_vec_list[j]
            pred_vec = pred_vec_list[j]
            # Gets average RMS and 1-r^2 error values for this image
            rms_err_val = get_error(3, True, pred_vec, ms_vec)
            rsq_val = 1 - r2_score(ms_vec, pred_vec)
            im_out = ims_in_fold[j]

            # Adds error values to appropriate lists based on crystallinity
            # of particle.
            # Image indices corresponding with the three crystallinity classes
            # are different in dataset 3 than in datasets 4 and 5.
            if thisrun.datanum == 3:
                if im_out in [0, 1, 2, 4, 5, 7, 8]:
                    amorph_rms.append(rms_err_val)
                    amorph_rsq.append(rsq_val)
                elif im_out in [9, 10, 11, 12, 13, 14]:
                    mixed_rms.append(rms_err_val)
                    mixed_rsq.append(rsq_val)
                elif im_out in [3, 6, 15, 16, 17, 18, 19]:
                    xtal_rms.append(rms_err_val)
                    xtal_rsq.append(rsq_val)

            elif thisrun.datanum in [4, 5]:
                if im_out in [0, 1, 3, 4, 5, 16, 19]:
                    amorph_rms.append(rms_err_val)
                    amorph_rsq.append(rsq_val)
                elif im_out in [6, 7, 8, 9, 10, 11]:
                    mixed_rms.append(rms_err_val)
                    mixed_rsq.append(rsq_val)
                elif im_out in [2, 12, 13, 14, 15, 17, 18]:
                    xtal_rms.append(rms_err_val)
                    xtal_rsq.append(rsq_val)

            # Creates error map plot for each image in each fold
            dif_image = dif_im_list[j]
            ms_image = ms_im_list[j]
            pred_image = pred_im_list[j]
            error_maps(dif_image, ms_image, pred_image, in_order_ims[j],
                       thisrun, fold=i, show_pred=True, show_zoom=True,
                       show_no_zoom=True, save=thisrun.save)

        # Adds error information to RunSpecs object
        thisrun.add_crystallity_error(amorph_rms, mixed_rms, xtal_rms,
                                      amorph_rsq, mixed_rsq, xtal_rsq)

    return


def error_maps(dif_image, actual_image, pred_image, im_to_show, thisrun,
               fold, show_pred=False, show_zoom=False, show_no_zoom=False,
               save=True, max_color_1=None, min_color_1=None):
    """
    Creates a figure with up to four subplots showing simulated images, errors

    Subplot 1: The actual (target) multislice image.
    Subplot 2: The image predicted by the model.
    Subplot 3: A map of the differences between the actual and predicted
        image, plotted on the same scale as those two images.
    Subplot 4: The same error map as Subplot 3, plotted on a reduced scale
        to allow better visualization of error.

    Parameters
    ----------
    dif_image : 2D array
        Error map, ie, the predicted image minus the actual image, pixel-wise.

    actual_image : 2D array
        The actual (target) image,

    pred_image : 2D array
        The image predicted by the model

    im_to_show : int
        The index of the image

    thisrun : RunSpecs object
        Contains all raw data and model parameters

    fold : int
        The index of the fold these images are part of

    show_pred : boolean (default False)
        Indicates whether to show predicted image

    show_zoom : boolean (default False)
        Indicates whether to show error map on a condensed scale

    show_no_zoom : boolean (default False)
        Indicates whether to show error map on the scale of the actual and
        predicted images

    save : boolean (default True)
        Indicates whether to save figure as .png file

    max_color_1 : float (default None)
        The maximum value on the color scale for the actual, predicted, and
        not-zoomed error map images. If none, it is set in the method using the
        maximum value from the actual and predicted images

    min_color_1 : float (default None)
        The minimum value on the color scale for the actual, predicted, and
        not-zoomed error map images. If none, it is set to 0.

    Returns
    -------
    None"""

    plt.close('all')
    frac = round((1-thisrun.fracin)*100)
    dn = thisrun.datanum
    model = thisrun.modeltype
    inputs = thisrun.inputs
    deg = thisrun.deg
    number = thisrun.number

    if max_color_1 is None:
        max_color_1 = max([np.amax(actual_image), np.amax(pred_image)])
    if min_color_1 is None:
        min_color_1 = 0
    tick_range_1 = np.linspace(min_color_1, max_color_1, num=5)

    # Sets scale for the zoomed-in error map separately from other 3 images
    if show_zoom is True:
        max_color_2 = np.amax(dif_image)
        min_color_2 = np.amin(dif_image)
        tick_range_2 = np.linspace(min_color_2, max_color_2, num=5)

    # Calculates total number of subplots
    total_plts = 1
    if show_pred:
        total_plts += 1
    if show_zoom:
        total_plts += 1
    if show_no_zoom:
        total_plts += 1

    if dn in [1, 3]:
        dname = 'Pt'
    elif dn == 4:
        dname = 'PtMo 5%'
    elif dn == 5:
        dname = 'PtMo 50%'

    # Initializes figure (size based on number of subplots total)
    n = 1
    if total_plts == 1:
        grid_row = 1
        grid_col = 1
        if dn == 1:
            fig = plt.figure(figsize=(3, 3))
        if dn in [3, 4, 5]:
            fig = plt.figure(figsize=(4, 4.5))

    elif total_plts == 2:
        grid_row = 1
        grid_col = 2
        if dn == 1:
            fig = plt.figure(figsize=(5, 3))
        if dn in [3, 4, 5]:
            fig = plt.figure(figsize=(10, 3.5))

    elif total_plts in [3, 4]:
        grid_row = 2
        grid_col = 2
        if dn == 1:
            fig = plt.figure(figsize=(10, 8))
        if dn in [3, 4, 5]:
            fig = plt.figure(figsize=(10, 8))

    # Plots the actual image (plot always included)
    image_colorscale_subplot(actual_image, 'Actual',
                             fig, grid_row, grid_col, n, min_color_1,
                             max_color_1, False, tick_range_1)
    plt.suptitle(dname+', degree ' + str(deg) + ', inputs ' + str(inputs) +
                 ', model ' + str(model) + ', '+str(frac) + '% out, slice ' +
                 str(im_to_show))

    # Plots the predicted image
    if show_pred:
        n = n + 1
        image_colorscale_subplot(pred_image, 'Predicted', fig, grid_row,
                                 grid_col, n, min_color_1, max_color_1,
                                 False, tick_range_1)

    # Plots the error map on the same scale as the actual and predicted images
    if show_no_zoom:
        n = n + 1
        image_colorscale_subplot(dif_image, 'Differences (large scale)',
                                 fig, grid_row, grid_col, n, min_color_1,
                                 max_color_1, False, tick_range_1)

    # Plots error map on its own scale
    if show_zoom:
        n = n + 1
        image_colorscale_subplot(dif_image, 'Differences (red. scale)',
                                 fig, grid_row, grid_col, n, min_color_2,
                                 max_color_2, False, tick_range_2)

    # Saves plot with identifying information in title
    if save:
        saveplot('d_dn' + str(dn) + '_n' + str(number) + '_s' +
                 str(im_to_show))
    else:
        plt.show()


def image_colorscale_subplot(image, subtitle, figure, grid_row, grid_col, n,
                             vmin, vmax, axes_labels_on, tick_range=False):
    """ Creates error map subplots for inclusion in larger error map figures

    Parameters
    ----------
    image : 2D array
        Data to plot

    subtitle : string
        Title for subplot

    figure : figure object

    grid_row : int
        How many rows of subplots will be in the figure

    grid_col : int
        How mant columns of subplots will be in the figure

    n : int
        Index of this subplot in the figure

    vmin : float
        The minimum value on the color scale

    vmax : float
        The maximum value on the color scale

    axes_labels_on : boolean
        Indicates whether to include tickmarks on axes that indicate number
        of pixels

    tick_range : list of floats, or boolean (default False)
        If a list, forces the tick marks on the color bar to be set at the
        specified values.
        If False, figure.colorbar sets locations of colorbar tickmarks.

    Returns
    -------
    None"""
    ax = figure.add_subplot(grid_row, grid_col, n)
    ax.set_aspect('equal', 'box')
    if tick_range is False:
        figure.colorbar(ax.pcolormesh(image, cmap=cm.viridis, vmin=vmin,
                                      vmax=vmax))
    else:
        tick_labels = []
        for i in range(len(tick_range)):
            tick_labels.append(round_scinote(tick_range[i], 1))
        cbar = figure.colorbar(ax.pcolormesh(image, cmap=cm.viridis, vmin=vmin,
                                             vmax=vmax), ticks=tick_range)
        cbar.ax.set_yticklabels(tick_labels)

    if axes_labels_on is False:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    else:
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])

    ax.set_xlabel(subtitle)
    return


def round_scinote(x, sf):
    """
    Rounds float to a number of significant digits, expresses in sci. notation

    Used in plotting to format how colorbar ticks are displayed.

    Parameters
    ----------
    x : float
        Number to be rounded

    sf : int
        Number of significant figures to round to

    Returns
    -------
    0 if x is 0, otherwise given number rounded to specified number of
    significant figures and expressed in scientific notation."""
    if x != 0:
        d = '%.'+str(sf)+'E'
        return d % Decimal(x)
    else:
        return 0


def parity(thisrun):
    """
    Creates parity plot using all test data.

    Parameters
    ----------
    thisrun : RunSpecs object
        Contains data for the run (including predicted information; note this
        means this method can only be run after the model has been fitted and
        applied to test data). Also contains all model parameters.

    Returns
    -------
    None"""
    predicted = thisrun.all_predicted
    ms_test = thisrun.all_ms_test
    dn = thisrun.datanum
    deg = thisrun.deg
    inputs = thisrun.inputs
    model = thisrun.modeltype
    frac = 1 - thisrun.fracin
    number = thisrun.number

    plt.close('all')
    frac = round(frac*100)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal', 'box')
    ax.plot(predicted, ms_test, 'k,')

    tick_range = []
    if dn in [1, 2, 4]:
        plt.plot([0, .2], [0, .2], color='r', linestyle='-')
        tick_range = [0, .05, .1, .15, .2]
        tick_labels = ['0', '.05', '.10', '.15', '20']
    elif dn == 3:
        ax.plot([0, .15], [0, .15], color='r', linestyle='-')
        tick_range = [0, .05, .1, .15]
        tick_labels = ['0', '.05', '.10', '.15']
    elif dn == 5:
        plt.plot([0, .1], [0, .1], color='r', linestyle='-')
        tick_range = [0, .025, .05, .075, .1]
        tick_labels = ['0', '.025', '.05', '.075', '.10']

#    tick_labels = []
#    for i in range(len(tick_range)):
#        tick_labels.append(round_scinote(tick_range[i], 1))

#    plt.title('Data=' + str(dn) + ', degree=' + str(deg) + ', inputs=' +
#              str(inputs) + ', model=' + str(model) + ', ' + str(frac) +
#              '% out parity plot')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks(tick_range)
    ax.set_yticks(tick_range)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
#    ax.tight_layout()

    if thisrun.save is True:
        saveplot('p_dn'+str(dn)+'_n'+str(number))
    else:
        plt.show()
    return


def get_error(errortype, pct, predicted, ms_test):
    """
    Returns average error of a specified type, optionally as a percent.

    Gets error for the test set given predicted and actual values. Can retrieve
    mean error, mean absolute error, or root-mean-square error, expressed as an
    actual value or as a percent (if percent, value is divided by the mean
    of the vector of actual values and multiplied by 100).

    Parameters
    ----------
    errortype : int
        Can be 1 (mean absolute error), 2 (mean error), or 3 (root-mean-square
        error)
        If not 1, 2, or 3, return -1

    pct : boolean
        If true, return the requested error type as a percent
        If false, return the actual error value

    predicted : array, shape = [n_test_pixels]
        The values predicted by the model

    ms_test : array, shape = [n_test_pixels]
        The actual values for the test set

    Returns
    -------
    av_pcterror : float
        if pct is True. Average value of the requested error type expressed
        as a percent.

    av_error : float
        if pct is False. Average value of the requested error type."""
    if errortype == 1:
        av_error, av_pcterror = mean_abs_error(predicted, ms_test)

    elif errortype == 2:
        av_error, av_pcterror = mean_error(predicted, ms_test)

    elif errortype == 3:
        av_error, av_pcterror = rms(predicted, ms_test)

    else:
        return -1

    if pct is True:
        return av_pcterror
    else:
        return av_error


def mean_abs_error(predicted, ms_data):
    """
    Calculates and returns average mean absolute error, as value and percent.

    To express as a percent, the error value is divided by the average of the
    vector of actual values and multiplied by 100

    Parameters
    ----------
    predicted : array, shape = [n_test_pixels]
        The values predicted by the model for the test set.

    ms_data : array, shape = [n_test_pixels]
        The actual values for the test set.

    Returns
    -------
    av_error : float
        average mean absolute error for the test set

    av_pcterror : float
        average mean absolute error for the test set, expressed as a percent.
    """
    error = []
    av_ms = mean(ms_data)
    points = len(predicted)
    for i in range(0, points):
        error.append(mean(abs(predicted[i] - ms_data[i])))
    av_error = mean(error)
    pcterror = error/av_ms*100
    av_pcterror = mean(pcterror)

    return av_error, av_pcterror


def mean_error(predicted, ms_data):
    """
    Calculates and returns average mean error, as value and percent.

    To express as a percent, the error value is divided by the average of the
    vector of actual values and multiplied by 100

    Parameters
    ----------
    predicted : array, shape = [n_test_pixels]
        The values predicted by the model for the test set.

    ms_data : array, shape = [n_test_pixels]
        The actual values for the test set.

    Returns
    -------
    av_error : float
        average mean error for the test set

    av_pcterror : float
        average mean error for the test set, expressed as a percent."""
    error = []
    av_ms = mean(ms_data)
    points = len(predicted)
    for i in range(0, points):
        error.append(mean(predicted[i] - ms_data[i]))
    av_error = mean(error)
    pcterror = error/av_ms*100
    av_pcterror = mean(pcterror)

    return av_error, av_pcterror


def rms(predicted, ms_data):
    """
    Calculates and returns root mean square (RMS) error, as value and percent.

    To express as a percent, the error value is divided by the average of the
    vector of actual values and multiplied by 100

    Parameters
    ----------

    predicted : array, shape = [n_test_pixels]
        The values predicted by the model for the test set.

    ms_data : array, shape = [n_test_pixels]
        The actual values for the test set.

    Returns
    -------

    av_error : float
        average RMS error for the test set

    av_pcterror : float
        average RMS error for the test set, expressed as a percent."""
    error = []
    av_ms = mean(ms_data)
    points = len(predicted)
    for i in range(0, points):
        error.append(square(predicted[i]-ms_data[i]))
    av_error = sqrt(mean(error))
    av_pcterror = av_error/av_ms*100

    return av_error, av_pcterror


def pct_rms_indiv(predicted, ms_data):
    """
    Calculates average percent RMS error, calculated differently than in rms.

    Calculates percent RMS error at each pixel using the actual value at just
    that pixel to convert to a percent, then takes an average. This
    is different from the rms method, which calculates actual RMS error at
    each pixel, takes an average, and then converts to a percent by dividing by
    the average actual value over the entire image.

    Parameters
    ----------
    predicted : array, shape = [n_test_pixels]
        The values predicted by the model for the test set.

    ms_data : array, shape = [n_test_pixels]
        The actual values for the test set.

    Returns
    -------
    The average percent RMS error, calculated pixel-by-pixel"""
    error = []
    points = len(predicted)

    for i in range(0, points):
        error.append(square((predicted[i]-ms_data[i])/ms_data[i]))

    return sqrt(mean(error))*100


def vector_to_image(data, shape):
    """
    Takes a 1D vector and transforms it into a 3D array of specified shape.

    Note that length of input data vector must equal total number of elements
    in specified 3D array.

    This is different than np.reshape() because it is specific to the structure
    of the image arrays we are dealing with. The indexing is a little
    different than np.reshape() would expect.

    Parameters
    ----------
    data : 1D array
        The data to be transformed

    shape : list, shape = [3]
        The shape the data will be transformed to.

    Returns
    -------
    image : 3D array
        The resulting 3D array, which represents a set of images."""
    image = np.zeros(shape)
    count = 0
    for k in range(0, shape[2]):
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                image[i, j, k] = data[count]
                count = count + 1

    return image


def get_ims(predicted, ms_data_trimmed, ims_to_show, shapes, inputs):
    """Rearranges 1D error vectors into 2D images and gets error maps

    Parameters
    ----------
    predicted : 1D array, shape = [n_test_pixels]
        The values predicted by the model

    ms_data_trimmed : 1D array, shape = [n_test_pixels]
        The actual (target) values for the test set (with some number of outer
        pixels cropped out, number depends on how many model inputs there are)

    ims_to_show : list of ints
        Indices of the desired images in the whole dataset

    shapes : list of lists with shape = [2]
        A list of the shapes of the desired images (each has a different shape
        which is why this isn't a single 2-element list)

    inputs : int
        Number of input pixels into the model.

    Returns
    -------
    dif_image_list : list of 2D arrays
        A list of the error map images (predicted minus actual)

    actual_image_list : list of 2D arrays
        A list of the actual images

    predicted_image_list
        A list of the images predicted by the model"""

    trimrows = math.floor(np.sqrt(inputs)/2)
    dif_image_list = []
    actual_image_list = []
    predicted_image_list = []

    # For each desired image, rearrange vectors into arrays of specified shape
    for n in range(len(ims_to_show)):
        shape = shapes[n]
        elements = (shape[0]-2*trimrows)*(shape[1]-2*trimrows)
        predicted_image = np.reshape(predicted[0:elements], (shape[0] - 2 *
                                     trimrows, shape[1] - 2 * trimrows))
        actual_image = np.reshape(ms_data_trimmed[0:elements],
                                  (shape[0] - 2 * trimrows,
                                   shape[1] - 2 * trimrows))
        predicted = np.delete(predicted, range(elements))
        ms_data_trimmed = np.delete(ms_data_trimmed, range(elements))

        # Obtain error map
        dif_image = predicted_image - actual_image

        dif_image_list.append(dif_image)
        actual_image_list.append(actual_image)
        predicted_image_list.append(predicted_image)

    return dif_image_list, actual_image_list, predicted_image_list


def get_im_vecs(predicted, ms_data_trimmed, ims_to_show, shapes, inputs):
    """Extracts vectors for each image from the data vectors for the test set

    Parameters
    ----------
    predicted : 1D vector, shape = [n_test_pixels]
        The vector of values predicted by the model for the whole test set

    ms_data_trimmed : 1D vector, shape = [n_test_pixels]
        The vector of actual values for the whole test set

    ims_to_show : list of ints
        The indices of the images in the test set

    shapes : list of lists with shape = [2]
        The shapes of the images in the test set, in the same order as
        ims_to_show

    inputs : int
        The number of model inputs

    Returns
    -------
    dif_vector_list : list of 1D vectors
        The list of error vectors for each image

    actual_vector_list : list of 1D vectors
        The list of vectors of actual values for each image

    predicted_vector_list : list of 1D vectors
        The list of vectors of values predicted by the model for each image
    """

    trimrows = math.floor(np.sqrt(inputs)/2)
    dif_vector_list = []
    actual_vector_list = []
    predicted_vector_list = []

    # Break large vector up into smaller vectors, one for each image
    for n in range(len(ims_to_show)):
        shape = shapes[n]
        elements = (shape[0]-2*trimrows)*(shape[1]-2*trimrows)
        predicted_vector = np.ravel(predicted[0:elements])
        actual_vector = np.ravel(ms_data_trimmed[0:elements])
        predicted = np.delete(predicted, range(elements))
        ms_data_trimmed = np.delete(ms_data_trimmed, range(elements))

        # Obtain difference vector
        dif_vector = predicted_vector - actual_vector

        dif_vector_list.append(dif_vector)
        actual_vector_list.append(actual_vector)
        predicted_vector_list.append(predicted_vector)

    return dif_vector_list, actual_vector_list, predicted_vector_list


def saveplot(name):
    """
    Saves current figure to the folder on the Python path

    Adds a timestamp to the given base name and saves to that updated name.

    Parameters
    ----------
    name : string
        The base name to save the plot under

    Returns
    -------
    None"""

    date = datetime.now()
    filedate = date.strftime(name + '_%Y-%m-%d_%H;%M;%S')
    plt.savefig(filedate + '.png', dpi=200, pad_inches=.2)
    return


def errorhist(error, bins, binmin, binmax, errortype, thisrun):
    """Creates and saves a histogram from the given error vector

    Parameters
    ----------
    error : 1D vector
        Error values for an image, test set, or series of test sets

    bins : int
        Number of bins to sort values into

    binmin : float
        Minimum error value at which to put a bin

    binmax : float
        Maximum error value at which to put a bin

    errortype : int
        Indicates which type of error is represented by error vector. Options
        are 1 (mean absolute error), 2 (mean error), 3 (RMS error), or 4
        (1-r^2). Used for labeling plot only.

    thisrun : RunSpecs object
        Contains model parameters used for labeling plot

    Returns
    -------
    None"""

    deg = thisrun.deg
    inputs = thisrun.inputs
    valtype = thisrun.cvtype
    modeltype = thisrun.modeltype
    fracin = thisrun.fracin
    save = thisrun.save
    plt.clf()
    plt.hist(error, bins=bins, range=(binmin, binmax))

    val = 'Crossval type: '
    if valtype == 1:
        val = val + 'image k fold, in order'
    elif valtype == 2:
        val = val + 'random block'
    elif valtype == 3:
        val = val + 'random point'
    elif valtype == 4:
        val = val + 'gridded random block'
    elif valtype == 5:
        val = val + 'image k fold, random order'

    if modeltype == 1:
        mod = ('Polynomial, degree: ' + str(deg) + ' inputs: ' + str(inputs) +
               ' wo cross')
    elif modeltype == 2:
        mod = ('Polynomial, degree: ' + str(deg) + ' inputs: ' + str(inputs) +
               ' w cross')
    elif modeltype == 3:
        mod = 'Ridge regression, inputs: ' + str(inputs)
    elif modeltype == 4:
        mod = 'Kernel ridge regression, inputs: ' + str(inputs)

    if errortype == 1:
        et = 'Mean absolute error, '
    elif errortype == 2:
        et = 'Mean error,  '
    elif errortype == 3:
        et = 'RMS error, '
    elif errortype == 4:
        et = '1 - R^2'

    plt.title(et + mod + '\n' + val + ', fraction in: ' + str(fracin))
    plt.xlabel('Error')
    plt.ylabel('Frequency')

    if save is True:
        saveplot('errorhist' + str(errortype))
    else:
        plt.show()
    return


def nCk(n, k):
    """Calculates number of possible combinations, 'n choose k'

    Parameters
    ----------
    n : int
    k : int

    Returns
    -------
    The number of possible combinations of k objects from a set of n total
    objects."""
    return int(reduce(mul, (Fraction(n-i, i+1) for i in range(k)), 1))
