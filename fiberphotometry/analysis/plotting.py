import matplotlib.pyplot as plt
import pandas as pd

from fcutils.plotting.utils import create_figure
from fcutils.plotting.colors import salmon, goldenrod

from fiberphotometry.variables import blue_color, violet_color
from fiberphotometry.analysis.utils import get_ldr_channel


def plot_traces_df(traces, signal_trace=None,  max_x = 10000):
    """
        Plot traces from a dataframe
    """
    # Get number of plots and crate fig
    columns = traces.columns
    if signal_trace is None: 
        n_plots = len(columns)
    else:
        n_plots = len(columns)+1
    f, axarr = create_figure(subplots=True, nrows=n_plots, figsize=(20, 16), sharex=True)

    # Plot individual traces
    for ax, col in zip(axarr, columns):
        if 'blue' in col.lower():
            color = blue_color
        elif 'violet' in col.lower():
            color = violet_color
        else:
            color = salmon
            
        ax.plot(traces[col][:max_x], color=col)
        ax.set(title=col)

    # Plot signal trace
    if signal_trace is not None:
        axes[-1].plot(signal_trace, color=goldenrod, lw=2)
        axes[-1].set(title='Signal')

    plt.show()



if __name__ == '__main__':
    df = "Z:\\swc\\branco\\rig_photometry\\tests\\200205\\200205_mantis_test_longer_exposure_noaudio\\FP_calcium_and_behaviour(0)-FP_calcium_camera_traces.hdf"
    
    inputs_file = ""
    inputs_ch = ""
    signal = get_ldr_channel(inputs_file, inputs_ch)
    
    plot_traces_df(pd.read_hdf(df, key='hdf'), signal_trace=signal)

