import sys
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')

import seaborn as sns
from scipy.optimize import curve_fit
import random
import numpy as np
import pandas as pd

from analysis_utils import setup, get_stimuli_from_ldr
from analysis_utils import blueled, blue_dff_color, violetled, motion_color, ldr_color

from fcutils.file_io.utils import check_file_exists
from fcutils.plotting.plotting_utils import *
from fcutils.plotting.colors import desaturate_color
from fcutils.plotting.plot_elements import plot_shaded_withline, ortholines

def double_exponential(x, a, b, c, d):
    # return a ** (b ** x )
    return a * np.exp(b * x) + c * np.exp(d*x)


def plot_effect_double_exponential(folder):
    def func(x, y):
        popt, pcov = curve_fit(double_exponential, x, y, maxfev=2000, 
                            p0=(1.0,  -1e-6, 1.0,  -1e-6),
                            bounds = [[1, -1e-1, 1, -1e-1], [100, 0, 100, 0]])

        y_pred = double_exponential(x, *popt)
        y_corr = y - y_pred
        return y, y_pred, y_corr, popt


    # Get data
    files, outpath, data, n_fibers = setup(folder, "effect_double_exp.png", True)
    if data is None: return

    blue = data.ch_0_signal.values[2:]
    violet = data.ch_0_motion.values[2:]
    x = np.arange(len(blue))

    # Fit double exponential
    blue, blue_pred, blue_corr, blue_popt = func(x, blue)
    violet, violet_pred, violet_corr, violet_popt = func(x, violet)

    # Plot
    f, axarr = create_figure(subplots=True, nrows=2, ncols=3, sharex=True, figsize=(20, 10))

    axarr[0].plot(blue, lw=3, color=blueled)
    axarr[0].plot(blue_pred, color='red')
    axarr[0].set(title='Raw and fit {} {} {} {}'.format(*[round(p, 4) for p in blue_popt]))

    axarr[1].plot(blue_corr, color=blueled)
    axarr[1].set(title='Raw - double exponential')

    axarr[2].plot((blue_corr-np.nanmedian(blue_corr)/np.nanmedian(blue_corr)), color=blueled)
    axarr[2].set(title='$\\frac{\\Delta f}{f}$')

    axarr[3].plot(violet, lw=3, color=violetled)
    axarr[3].plot(violet_pred, color='red')
    axarr[3].set(title='Raw and fit {} {} {} {}'.format(*[round(p, 4) for p in violet_popt]))
    axarr[4].plot(violet_corr, color=violetled)
    axarr[5].plot((violet_corr-np.nanmedian(violet_corr)/np.nanmedian(violet_corr)), color=violetled)

    sns.despine(offset=2, trim=True)
    save_figure(f, outpath.split(".png")[0])
    close_figure(f)


if __name__ == "__main__":
    folder = '/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/fiberphot_data/200124_fiberp/240120_id_994381_freelymoving_visualstim'
    plot_effect_double_exponential(folder)