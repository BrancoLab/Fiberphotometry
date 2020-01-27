import sys
sys.path.append('./')

import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit


from fcutils.file_io.utils import listdir, check_file_exists, check_create_folder
from fcutils.plotting.colors import *
from fcutils.maths.filtering import median_filter_1d

# ? Define a bunch of colors
blueled = lightskyblue
blue_dff_color = plum
violetled = violet
motion_color = thistle
ldr_color = salmon


def get_files_in_folder(folder):
    """
        Given the path to an experiment's folder it finds the correct files within and returns them as a dictionary
    """
    files = listdir(folder)

    try:
        behavcam = [f for f in files if 'cam1' in f][0]
    except:
        behavcam = None
    
    try:
        cacam = [f for f in files if 'cam0' in f][0]
    except:
        cacam = None
        
    sensors = [f for f in files if 'sensors_data.csv' in f][0]
    analysis = os.path.join(folder, 'analysis')
    check_create_folder(analysis)


    return dict(behaviour=behavcam, calcium=cacam, sensors=sensors, analysis=analysis)




def get_data_from_sensors_csv(sensors_file, fit_range = [], invert=False):
    """
        Given a sensors_data.csv it loads the data and takes care of the linear regression

        :param invert: bool, for the first few recordings blue and violed LED signals were mixed up.
                    use invert=True to correct this.
    """

    def double_exponential(x, a, b, c, d):
        return a * np.exp(b * x) + c * np.exp(d*x)

    def remove_exponential(x, y):
        """ Fits a double exponential to the data and returns the results """
        popt, pcov = curve_fit(double_exponential, x, y, maxfev=2000, 
                            p0=(1.0,  -1e-6, 1.0,  -1e-6),
                            bounds = [[1, -1e-1, 1, -1e-1], [100, 0, 100, 0]])

        fitted_doubleexp = double_exponential(x, *popt)
        y_pred = y - (fitted_doubleexp - np.min(fitted_doubleexp))
        return y_pred

    # Load data
    data = pd.read_csv(sensors_file)
    n_fibers = len([c for c in data.columns if 'ch_' in c and 'signal' in c])

    # For each fiber get the regressed signal and df/f
    for n in range(n_fibers):
        if not invert:
            blue = data['ch_{}_signal'.format(n)].values[2:]
            violet = data['ch_{}_motion'.format(n)].values[2:]
        else:
            violet = data['ch_{}_signal'.format(n)].values[2:]
            blue = data['ch_{}_motion'.format(n)].values[2:]
        
        if fit_range:
            if len(fit_range) == 1:
                blue = data['ch_{}_signal'.format(n)].values[2+fit_range[0]:]
                violet = data['ch_{}_motion'.format(n)].values[2+fit_range[0]:] 
            elif len(fit_range) == 2:
                blue = data['ch_{}_signal'.format(n)].values[2+fit_range[0]:2+fit_range[1]]
                violet = data['ch_{}_motion'.format(n)].values[2+fit_range[0]:2+fit_range[1]]                
                
        # Correct bleaching for each signal
        x = np.arange(len(blue))
        blue_no_bleach = remove_exponential(x, blue)
        violet_no_bleach = remove_exponential(x, violet)

        # compute dff for each led
        blue_bsl = np.nanpercentile(blue_no_bleach,5)
        violet_bsl = np.nanpercentile(violet_no_bleach,5)
        blue_dff = (blue_no_bleach - blue_bsl)/blue_bsl
        violet_dff = (violet_no_bleach - violet_bsl)/violet_bsl

        # regress signal
        regressor = LinearRegression()  
        regressor.fit(violet_dff.reshape(-1, 1), blue_dff.reshape(-1, 1))
        expected_blue_dff = violet_dff*regressor.coef_[0][0] + regressor.intercept_[0]
        corrected_blue_dff = blue_dff - expected_blue_dff
        
        # if fitting was not done on whole trace, pad back to same length with nans
        if fit_range != []:
            pad_bef = fit_range[0]
            if len(fit_range) == 1:
                pad_aft = 0
            else:
                pad_aft = len(data['ch_{}_signal'.format(n)].values[2:])-fit_range[1]-fit_range[0]
                
            blue_no_bleach = np.pad(blue_no_bleach, (pad_bef,pad_aft), 'constant', constant_values = np.nan)
            violet_no_bleach = np.pad(violet_no_bleach, (pad_bef,pad_aft), 'constant', constant_values = np.nan)
            blue_dff = np.pad(blue_dff, (pad_bef,pad_aft), 'constant', constant_values = np.nan)
            violet_dff = np.pad(violet_dff, (pad_bef,pad_aft), 'constant', constant_values = np.nan)
            corrected_blue_dff = np.pad(corrected_blue_dff, (pad_bef,pad_aft), 'constant', constant_values = np.nan)
     

        # Add stuff to the dataframe
        data['ch_{}_blue_nobleach'.format(n)] = np.concatenate([[0, 0], blue_no_bleach])
        data['ch_{}_violet_nobleach'.format(n)] = np.concatenate([[0, 0], violet_no_bleach])
        data['ch_{}_blue_dff'.format(n)] = np.concatenate([[0, 0], blue_dff])
        data['ch_{}_violet_dff'.format(n)] = np.concatenate([[0, 0], violet_dff])
        data['ch_{}_corrected'.format(n)] = np.concatenate([[0, 0], corrected_blue_dff])
    return data, n_fibers


def get_stimuli_from_ldr(ldr, th=4.45):
    " detects onset and offset of stimuli from the ldr signal"
    ldr_copy = np.zeros_like(ldr)
    ldr_copy[ldr > th] = 1

    ldr_onset = np.where(np.diff(ldr_copy) > .5)[0]
    ldr_offset = np.where(np.diff(ldr_copy) < -.5)[0]

    return ldr_onset, ldr_offset


def setup(folder, filename, overwrite, smooth_motion=True, **kwargs):
    """
        Gets data, checks if file exists..
    """
    # Get files
    files = get_files_in_folder(folder)

    # Get the name of the destination plot and check if it exists
    name = os.path.split(folder)[-1]
    outpath = "{}_{}".format(os.path.join(files['analysis'], name), filename)

    if check_file_exists(outpath) and not overwrite:
        return None, None, None, None

    if files['behaviour'] is None or files['calcium'] is None:
        return None, None, None, None
    
    if kwargs is not None and 'fit_range' in kwargs:
        if len(kwargs['fit_range'])>2: return None, None, None, None
    
    # Get sensors data and make sure everything is in place
    data, n_fibers = get_data_from_sensors_csv(files['sensors'], **kwargs)


    if 'behav_mvmt' not in data.columns or 'ldr' not in data.columns:
        return None, None, None, None

    # Smooth motion signal
    if smooth_motion:
        data['behav_mvmt'] = median_filter_1d(data['behav_mvmt'].values, pad=10, kernel=5)
    return files, outpath, data, n_fibers