import matplotlib.pyplot as plt
import pandas as pd

from fcutils.plotting.utils import create_figure
from fcutils.plotting.colors import salmon

from fiberphotometry.variables import blue_color, violet_color


def plot_traces_df(traces, max_x = 10000):
    """
        Plot traces from a dataframe
    """
    columns = traces.columns
    f, axarr = create_figure(subplots=True, nrows=len(columns), figsize=(20, 16), sharex=True)

    for ax, col in zip(axarr, columns):
        if 'blue' in col.lower():
            color = blue_color
        elif 'violet' in col.lower():
            color = violet_color
        else:
            color = salmon
            
        ax.plot(traces[col][:max_x], color=col)
        ax.set(title=col)

    plt.show()



if __name__ == '__main__':
    df = "Z:\\swc\\branco\\rig_photometry\\tests\\200205\\200205_mantis_test_longer_exposure_noaudio\\FP_calcium_and_behaviour(0)-FP_calcium_camera_traces.hdf"
    plot_traces_df(pd.read_hdf(df, key='hdf'))

