import os
import sys
try:
    import cupy as cp
    use_cupy = True
except:
    use_cupy = False

import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

sys.path.append('./')

from fcutils.file_io.utils import get_file_name, check_file_exists
from fcutils.video.utils import get_cap_from_file, get_cap_selected_frame, get_video_params, cap_set_frame

from fiberphotometry.analysis.utils import manually_define_rois, split_blue_violet_channels, get_ldr_channel

DEBUG = False
if DEBUG:
    plot_double_exp=False
    plot_correction=False
    plot_dff=True
    import matplotlib.pyplot as plt

class SignalExtraction:
    def __init__(self, video_path, n_rois=4, roi_radius=125, 
                    save_path=None, overwrite=False,
                    traces_file=None):

        # Open video and store a few related vars
        self.video_path = video_path
        self.video_name = get_file_name(video_path)
        self.video = get_cap_from_file(video_path)
        self.video_params = get_video_params(self.video) # nframes, width, height, fps
        self.first_frame = get_cap_selected_frame(self.video, 0)
        

        self.n_rois = n_rois
        self.roi_radius = roi_radius

        if save_path is not None:
            if 'hdf' not in save_path:
                raise ValueError("The save path should point to a hdf file")
            self.save_path = save_path
        else:
            self.save_path = self.video_path.split(".")[0]+"_traces.hdf"
        self.save_raw_path = self.video_path.split(".")[0]+"_raw_traces.hdf"
        self.overwrite = overwrite

        self.traces_file = traces_file
        if traces_file is not None:
            check_file_exists(traces_file, raise_error=True)
        
        self.ROI_masks = []


    # ---------------------------------------------------------------------------- #
    #                         EXTRACTING SIGNAL FROM VIDEO                         #
    # ---------------------------------------------------------------------------- #

    def get_ROI_masks(self, mode='manual'):
        """
            Gets the ROIs for a recording as a list of masked numpy arrays.

            :param frame: first frame of the video to analyse
            :param mode: str, manual for manual ROIs identification, else auto
        """
        if self.traces_file is not None:
            print("A traces file was passed, no need to extract ROI locations")
            return

        # Get ROIs
        if mode.lower() != 'manual':
            raise NotImplementedError
            # TODO work on extracting ROIs from a template image
        else:
            ROIs = manually_define_rois(self.first_frame, self.n_rois, self.roi_radius)

        # Get ROIs masks
        self.ROI_masks = []
        for roi in ROIs:
            blank = np.zeros_like(self.first_frame)
            cv2.circle(blank, (roi[0], roi[1]), self.roi_radius, (255, 255, 255), -1) 
            blank = (blank[:, :, 0]/255).astype(np.float64)
            blank[blank == 0] = np.nan
            self.ROI_masks.append(blank)

    def extract_signal(self):
        """
            Extract signal from a video of the fiber bundle. It will need to use ROi masks, so run 
            get_ROI_masks first. For each frame and each mask the frame is multiplied by the mask 
            to get just the pixels in the ROI and the average of these is taken as the signal for that
            ROI at that frame. Results are saved as a pandas dataframe (as a .h5 file)
        """

        if check_file_exists(self.save_path) and not self.overwrite:
            print("A path with the results exists already, returning that")
            return pd.read_hdf(self.save_path, key='hdf')

        if not self.ROI_masks and self.traces_file is None: 
            print("No ROI masks were defined, please run get_ROI_masks first")
            return None

        if self.traces_file is None:
            # Reset cap
            cap_set_frame(self.video, 0)

            # Perpare arrays to hold the data
            traces = {k:np.zeros(self.video_params[0]) for k in range(self.n_rois)}

            # Extract
            if not DEBUG:
                for frame_n in tqdm(range(self.video_params[0])):
                    ret, frame = self.video.read()
                    if not ret: 
                        raise ValueError("Could not read frame {}".format(i))

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    for n, mask in enumerate(self.ROI_masks):
                        if use_cupy:
                            masked = (frame * mask).astype(cp.float64)
                            traces[n][frame_n] = cp.nanmean(masked.astype(cp.uint8))
                        else:
                            masked = (frame * mask).astype(np.float64)
                            traces[n][frame_n] = np.nanmean(masked.astype(np.uint8))

                traces = pd.DataFrame(traces)

                # Split blue and violet traces
                traces = split_blue_violet_channels(traces)
        else:
            traces = pd.read_hdf(self.traces_file)
        raw_traces = traces.copy()


        # Remove double exponential
        traces = self.remove_double_exponential(traces)

        # Regress violet from blue. 
        traces = self.regress_violet_from_blue(traces)

        # Compute DF/F
        traces = self.compute_dff(traces)
       
        # Save and return
        print("Extraction completed.")
        print("Saving  raw traces at: {}".format(self.save_raw_path))
        raw_traces.to_hdf(self.save_raw_path, key='hdf')
        print("Saving  processed traces at: {}".format(self.save_path))
        traces.to_hdf(self.save_path, key='hdf')
        return raw_traces, traces

    # ---------------------------------------------------------------------------- #
    #                                EXTRACTING DF/F                               #
    # ---------------------------------------------------------------------------- #
    @staticmethod
    def double_exponential(x, a, b, c, d):
        return a * np.exp(b * x) + c * np.exp(d*x)

    def remove_exponential_from_trace(self, x, y):
        """ Fits a double exponential to the data and returns the results
        
            :param x: np.array with time indices
            :param y: np.array with signal

            :returns: np.array with doble exponential corrected out
        """
        popt, pcov = curve_fit(self.double_exponential, x, y,
                            maxfev=2000, 
                            p0=(1.0,  -1e-6, 1.0,  -1e-6),
                            bounds = [[1, -1e-1, 1, -1e-1], [100, 0, 100, 0]])

        fitted_doubleexp = self.double_exponential(x, *popt)
        y_pred = y - (fitted_doubleexp - np.min(fitted_doubleexp))
        return y_pred, fitted_doubleexp

    def remove_double_exponential(self, traces):
        time = np.arange(len(traces))
        for column in traces.columns:
            before = traces[column].values.copy()

            traces[column], double_exp = self.remove_exponential_from_trace(time, traces[column].values)

            if DEBUG and plot_double_exp:
                plt.plot(before, color='green', lw=2, label='raw')
                plt.plot(traces[column].values, color='k', lw=.5, label='correct')
                plt.plot(double_exp, color='red', label='double exp')
                plt.legend()
                plt.show()

        return traces

    def compute_dff(self, traces):
        for column in traces.columns:
            trace = traces[column].values.copy()
            baseline = np.nanmedian(trace)
            traces[column] = (trace-baseline)/baseline

            if DEBUG and plot_dff:
                f, axarr = plt.subplots(nrows=2)
                axarr[0].plot(trace, color='k', label='trace')
                axarr[0].axhline(baseline, color='m', label='baseline')
                axarr[1].plot(traces[column], color='g', label='dff')
                for ax in axarr: ax.legend()
                plt.show()
        return traces

    def regress_violet_from_blue(self, traces):
        for roi_n in np.arange(self.n_rois):
            blue = traces[str(roi_n)+'_blue'].values
            violet = traces[str(roi_n)+'_violet'].values

            regressor = LinearRegression()  
            regressor.fit(violet.reshape(-1, 1), blue.reshape(-1, 1))
            expected_blue = regressor.predict(violet.reshape(-1, 1)).ravel()        
            corrected_blue = (blue - expected_blue)/expected_blue
            traces[str(roi_n)+'_dff'] = corrected_blue

            if DEBUG and plot_correction:
                f, axarr = plt.subplots(nrows=3)
                axarr[0].plot(blue, color='b', label='blue')
                axarr[1].plot(violet, color='m',  label='violet')
                axarr[1].plot(expected_blue, color='k',  label='expected_blue')
                axarr[2].plot(corrected_blue, color='red', label='corrected_blue')
                for ax in axarr: ax.legend()
                plt.show()
        return traces

    # ---------------------------------------------------------------------------- #
    #                                     MISC                                     #
    # ---------------------------------------------------------------------------- #
    def add_signal_to_traces(self, traces, signal_file, signal_channel):
        """
            Extracts a signal (e.g. LDR analog input) from a .tdms or .pd file
            and downsamples it to align it to the traces.

            :param traces: pd.DataFrame with traces
            :param signal_file: str, path to .tdms file
            :param signal_channel: str, name of the tdms channel to use 
        """
        print('Adding signal to traces')
        signal = get_ldr_channel(signal_file, ldr_channel=signal_channel,
                downsample_to=len(traces))
        traces[signal_channel] = signal

        print("Saving  updated processed traces at: {}".format(self.save_path))
        traces.to_hdf(self.save_path, key='hdf')
        return traces



if __name__ == "__main__":
    video = "Z:\\swc\\branco\\rig_photometry\\tests\\200205\\200205_mantis_test_longer_exposure_noaudio\\FP_calcium_and_behaviour(0)-FP_calcium_camera.mp4"


    se = SignalExtraction(video, overwrite=True, traces_file='test.h5')
    se.get_ROI_masks()
    raw, traces = se.extract_signal()

    signal_file =  r"Z:\swc\branco\rig_photometry\tests\200205\200205_mantis_test_longer_exposure_noaudio\FP_calcium_and_behaviour(0).tdms"
    se.add_signal_to_traces(traces, signal_file, 'FP_ldr_signal')




