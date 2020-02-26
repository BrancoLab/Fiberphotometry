import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.append("./")

from fcutils.plotting.utils import create_figure
from fcutils.plotting.colors import salmon, goldenrod

from fiberphotometry.variables import blue_color, violet_color
from fiberphotometry.analysis.utils import get_ldr_channel


def plot_traces_df(traces,   max_x = -1):
    """
        Plot traces from a dataframe
    """
    # Get number of plots and crate fig
    columns = traces.columns
    n_plots = len(columns)
    f, axarr = create_figure(subplots=True, nrows=n_plots, figsize=(20, 16), sharex=False)

    # Plot individual traces
    for ax, col in zip(axarr, columns):
        if 'blue' in col.lower():
            color = blue_color
        elif 'violet' in col.lower():
            color = violet_color
        elif 'dff' in col.lower():
            color = salmon
        else:
            color = goldenrod
            
        ax.plot(traces[col], color=color)
        ax.set(title=col)


    f.tight_layout()
    plt.show()



if __name__ == '__main__':
    df = "Z:\\swc\\branco\\rig_photometry\\tests\\200205\\200205_mantis_test_longer_exposure_noaudio\\FP_calcium_and_behaviour(0)-FP_calcium_camera_traces.hdf"
    

    plot_traces_df(pd.read_hdf(df, key='hdf'))

