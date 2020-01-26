import sys
sys.path.append('./')

import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

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


def get_data_from_sensors_csv(sensors_file, invert=False):
    """
        Given a sensors_data.csv it loads the data and takes care of the linear regression

        :param invert: bool, for the first few recordings blue and violed LED signals were mixed up.
                    use invert=True to correct this.
    """

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

        # compute dff for each led
        blue_dff = (blue - np.nanmedian(blue))/np.nanmedian(blue)
        violet_dff = (violet - np.nanmedian(violet))/np.nanmedian(violet)

        # regress signal
        regressor = LinearRegression()  
        regressor.fit(violet_dff.reshape(-1, 1), blue_dff.reshape(-1, 1))
        expected_blue_dff = violet_dff*regressor.coef_[0][0] + regressor.intercept_[0]
        corrected_blue_dff = blue_dff - expected_blue_dff
        corrected_blue_dff = np.concatenate([[0, 0], corrected_blue_dff])

        data['ch_{}_corrected'.format(n)] = corrected_blue_dff
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

    # Get sensors data and make sure everything is in place
    data, n_fibers = get_data_from_sensors_csv(files['sensors'], **kwargs)


    if 'behav_mvmt' not in data.columns or 'ldr' not in data.columns:
        return None, None, None, None

    # Smooth motion signal
    if smooth_motion:
        data['behav_mvmt'] = median_filter_1d(data['behav_mvmt'].values, pad=10, kernel=5)
    return files, outpath, data, n_fibers