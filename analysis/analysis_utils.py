import sys
sys.path.append('./')

import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

from utils.file_io import listdir

def get_files_in_folder(folder):
    """
        Given the path to an experiment's folder it finds the correct files within and returns them as a dictionary
    """
    files = listdir(folder)

    behavcam = [f for f in files if 'cam1' in f][0]
    cacam = [f for f in files if 'cam0' in f][0]
    sensors = [f for f in files if 'sensors_data.csv' in f][0]

    return dict(behaviour=behavcam, calcium=cacam, sensors=sensors)


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

        # regress signal
        regressor = LinearRegression()  
        regressor.fit(violet.reshape(-1, 1), blue.reshape(-1, 1))
        expected_blue = violet*regressor.coef_[0][0] + regressor.intercept_[0]
        corrected_blue = blue - expected_blue
        corrected_blue = np.concatenate([[0, 0], corrected_blue])

        data['ch_{}_corrected'.format(n)] = corrected_blue

        # get df/f
        chmean = np.mean(corrected_blue)
        data['ch_{}_dff'.format(n)] = (corrected_blue - chmean)/chmean

    return data, n_fibers
