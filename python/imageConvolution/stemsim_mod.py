#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main file that creates and assesses models based on input parameters.

Contains a method that creates and assesses models (run) and a class that
defines an object to store relevant run parameters and information (RunSpecs).

@author: Aidan Combs
@version: 3.0
"""

import fit_predict_mod
import output
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import r2_score

# modeltype 1: polynomial no crossterms no log
# modeltype 2: polynomial crossterms no log
# modeltype 3: ridge regression
# modeltype 4: kernel ridge regression
# modeltype 5: gaussian process regression
# modeltype 6: polynomial no crossterms log
# modeltype 7: polynomial crossterms log

# crossvaltype 1: slicewise, in order k fold
# crossvaltype 2: random blocks
# crossvaltype 3: random pixels, no buffer
# crossvaltype 4: grid blocks, k fold
# crossvaltype 5: slicewise, random k fold
# crossvaltype 6: slicewise, random leave-out-X

# errortype 1: mean absolute error
# errortype 2: mean error
# errortype 3: RMS error

# datanum 1: Zhongnan v0 data
# datanum 2: Zhongnan v1 data
# datanum 3: Pt data
# datanum 4: PtMo (5% Mo) data
# datanum 5: PtMo (50% Mo) data


def run(thisrun):
    """
    Passes RunSpecs object to the methods that train and assess models and
    report results.

    Parameters
    ----------
        thisrun : RunSpecs object
            contains data and model parameters for the run.

    Returns
    -------
        dataframe containing results
    """
    if thisrun.blurring is True:
        thisrun.conv_data = fit_predict_mod.blur(thisrun.conv_data)

    for rep in range(thisrun.repeats):
        thisrun.clear_predicted_list()
        # trains models according to given parameters, returns a list of models
        model_list = fit_predict_mod.build_train(thisrun)
        im_in_f_list = thisrun.im_in_f_list
        i = 0
        # gets and stores predicted values for each model
        for model in model_list:
            conv_test = thisrun.conv_test_list[i]
            ms_test = thisrun.ms_test_list[i]
            predicted, time = fit_predict_mod.predict(conv_test, model, thisrun)
            thisrun.add_test(predicted, ms_test, conv_test, time)
            i = i+1

        # If image-wise cross val, create difference plots and
        # find the error statistics for the different particle
        # compositions
        if thisrun.cvtype == 5 or thisrun.cvtype == 1 or thisrun.cvtype == 6:
            output.get_crystallity_and_error_maps(im_in_f_list, thisrun)
        else:
            thisrun.add_crystallity_error([-1], [-1], [-1], [-1], [-1], [-1])

    # Create parity plots (for all cross val types)--for all test data
    output.parity(thisrun)
    # summarize results in a csv
    thisrun.get_results_csv()

    if thisrun.save is True:
        date = datetime.now()
        filedate = date.strftime(thisrun.name + '_' + str(thisrun.number) +
                                 '_%Y-%m-%d_%H;%M;%S')
        thisrun.result.to_csv(filedate + '.csv', header=True, index=False)
    return thisrun.result


class RunSpecs:
    def __init__(self, datanum, fracin, deg, inp_side_len, cvtype, modeltype,
                 repeats, loxreps, blurring, blocksize, crop_pixels, name,
                 save, number):
        """
        Class that stores all pertinant information for the run.

        Parameters
        ----------
            datanum : integer indicating which dataset to use.
                      Results presented: 3 = Pt, 4 = PtMo 5%, 5 = PtMo50%
            fracin : percentage of the data to leave in the training set as a
                     decimal (ranges from 0-1)
            deg : degree of the polynomial to test
            inp_side_len : side length of the input pixel grid
            cvtype : type of cross validation to use.
                     Results presented: 3 = random pixel k-fold, 4 =
                     random block k-fold, 5 = slicewise k-fold, 6 = slicewise
                     leave-out-X.
            modeltype : form of model
                        Results presented: 1 = polynomial no cross terms no
                        log term, 2 = polynomial cross terms no log term,
                        6 = polynomial no cross terms with log term, 6 =
                        polynomial cross terms and log term
            repeats : number of times to repeat cross validation process
            loxreps : number of times to repeat the leave-out-X cross
                      validation
            blurring : whether or not to apply a Gaussian filter to the
                       convolution data before fitting models. False in all
                       results presented.
            blocksize : side length of the pixel block used in the random
                        block cross validation (7 in all results presented)
            crop_pixels : number of pixel rows to crop from each image edge
            name : the name of the run (used for saving data)
            save : whether or not to save results
            number : the line number of the run in the input csv file

        Returns
        -------
        RunSpecs object
        """
        self.c = ['number', 'fraction_in', 'test_cen', 'train_cen', 'out_num',
                  'all_test_num', 'degree', 'inputs', 'terms', 'blocksize',
                  'val_type', 'model_type', 'cropped_pixels',
                  'eval_time_per_pixel',
                  'eval_time_sd', 'average_fit_time_total',
                  'rms_pct', 'rms_pct_sem', 'rms', 'rms_sem', 'pct_imp',
                  '1-r^2', 'amorph_rms', 'mixed_rms', 'xtal_rms',
                  'amorph_1-r^2', 'mixed_1-r^2', 'xtal_1-r^2']
        self.datanum = datanum
        self.fracin = fracin
        self.deg = deg
        self.inp_side_len = inp_side_len
        self.inputs = self.inp_side_len ** 2
        self.cvtype = cvtype
        self.modeltype = modeltype
        self.repeats = repeats
        self.loxrep = loxreps
        self.blocksize = blocksize
        self.name = name
        self.number = number
        self.save = save
        self.blurring = blurring
        self.crop_pixels = crop_pixels

        self.predicted_list = []
        self.conv_test_list, self.ms_test_list = [], []
        self.conv_train_list, self.ms_train_list = [], []
        self.all_time_vec = []
        self.all_predicted, self.all_ms_test, self.all_conv_test = [], [], []
        self.amorph_rms, self.mixed_rms, self.xtal_rms = [], [], []
        self.amorph_rsq, self.mixed_rsq, self.xtal_rsq = [], [], []
        self.all_num = 0
        self.pct_ma_err, self.pct_rms, self.all_ma_err = [], [], []
        self.ma_err, self.rms, self.orig_error = [], [], []
        self.std_rms, self.std_pct_rms = [], []
        self.fit_time = []
        self.im_in_f_list = []

        self.termnums = fit_predict_mod.find_term_nums(self)
        self.trimrows = int((np.sqrt(self.inputs)-1)/2)
        self.ms_data, self.conv_data = fit_predict_mod.load_data(self.datanum,
                                                             self.crop_pixels)
        self.result = pd.DataFrame(columns=self.c)

    def set_train_test_out_nums(self, train_centers, test_centers, out_num):
        """
        Stores number of pixels in the test and training sets

        Parameters
        ----------
        train_centers : number of pixels in the multislice training set
        test_centers : number of pixels in the multislice test set
        out_num : number of pixels not in either the multislice training or
                  multislice test set (because random block cross val leaves
                  a buffer around blocks)

        Returns
        -------
        None
        """
        self.train_centers = train_centers
        self.test_centers = test_centers
        self.out_num = out_num
        self.train_centers_perc = (train_centers * 100 /
                                   (train_centers+test_centers+out_num))
        self.test_centers_perc = (test_centers * 100 /
                                  (train_centers+test_centers+out_num))
        self.out_num_perc = (out_num * 100 /
                             (train_centers+test_centers+out_num))

    def add_test(self, predicted, ms_test, conv_test, time):
        """
        Stores information for a model test.

        Parameters
        ----------
        predicted : list of pixel intensity values predicted by model
        ms_test : list pf multislice test set values
        conv_test : list of convolution test set values
        time : time taken to apply the model

        Returns
        -------
        None
        """
        self.all_predicted.extend(predicted)
        self.all_ms_test.extend(ms_test)
        self.all_conv_test.extend(conv_test)
        self.all_time_vec.append(time)
        self.all_num = self.all_num + len(predicted)

    def add_crystallity_error(self, amorph_rms, mixed_rms, xtal_rms,
                              amorph_rsq, mixed_rsq, xtal_rsq):
        """
        Stores crystallinity class-specific error information

        Parameters
        ----------
        amorph_rms : average root-mean-square error for amorphous particles
        mixed_rms : average root-mean-square error for mixed-crystallinity
                    particles
        xtal_rms : average root-mean-square error for crystalline particles
        amorph_rsq : average R squared value for amorphous particles
        mixed_rsq : average R squared value for mixed particles
        xtal_rsq : average R squared value for crystalline particles

        Returns
        -------
        None
        """
        self.amorph_rms.extend(amorph_rms)
        self.amorph_rsq.extend(amorph_rsq)
        self.mixed_rms.extend(mixed_rms)
        self.mixed_rsq.extend(mixed_rsq)
        self.xtal_rms.extend(xtal_rms)
        self.xtal_rsq.extend(xtal_rsq)

    def add_predicted(self, predicted):
        """
        Stores predicted pixel intensity information for a single model

        Parameters
        ----------
        predicted : list of predicted values for a model

        Returns
        -------
        None
        """
        self.predicted_list.append(predicted)

    def add_fit_time(self, fit_time):
        """
        Stores time taken to fit the model to the training data

        Parameters
        ----------
        fit_time : time taken to fit the model to the training data

        Returns
        -------
        None
        """
        self.fit_time.append(fit_time)

    def clear_predicted_list(self):
        """
        Clears prior predicted data from predicted_list

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.predicted_list = []

    def get_results_csv(self):
        """
        Writes results for the run to a dataframe and stores the dataframe

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for n in range(self.all_num):
            self.all_ma_err.append(abs(self.all_predicted[n] -
                                       self.all_ms_test[n]))
        std_rms = np.std(self.all_ma_err)/np.sqrt(self.all_num)
        std_pct_rms = (np.std(self.all_ma_err * 100 /
                              np.mean(self.all_ms_test))/np.sqrt(self.all_num))
        origerror = output.get_error(3, False, self.all_conv_test,
                                     self.all_ms_test)
        rms = output.get_error(3, False, self.all_predicted, self.all_ms_test)
        pctrms = output.get_error(3, True, self.all_predicted,
                                  self.all_ms_test)
        amrms = np.mean(self.amorph_rms)
        mixrms = np.mean(self.mixed_rms)
        xrms = np.mean(self.xtal_rms)
        amrsq = np.mean(self.amorph_rsq)
        mixrsq = np.mean(self.mixed_rsq)
        xrsq = np.mean(self.xtal_rsq)
        r_sq = r2_score(self.all_ms_test, self.all_predicted)
        time = np.mean(self.all_time_vec)
        timesd = np.std(self.all_time_vec)
        fit_time_av = np.mean(self.fit_time)

        thisrow = np.array([[self.number, self.fracin, self.test_centers,
                             self.train_centers, self.out_num, self.all_num,
                             self.deg, self.inputs, self.termnums,
                             self.blocksize, self.cvtype, self.modeltype,
                             self.crop_pixels, time, timesd, fit_time_av,
                             pctrms, std_pct_rms,
                             rms, std_rms, 1-rms/origerror, 1-r_sq,
                             amrms, mixrms, xrms, amrsq, mixrsq, xrsq]])
        thisrowdf = pd.DataFrame(data=thisrow, columns=self.c)
        self.result = self.result.append(thisrowdf, ignore_index=True)
