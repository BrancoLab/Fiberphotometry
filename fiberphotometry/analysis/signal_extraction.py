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

from fiberphotometry.analysis.utils import manually_define_rois, split_blue_violet_channels


class SignalExtraction:
    def __init__(self, video_path, n_rois=4, roi_radius=125, 
                    save_path=None, overwrite=False):

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
        self.save_raw_path = self.video_path.split(".")[0]+"_raw."+self.video_path.split(".")[1]
        self.overwrite = overwrite

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

        if not self.ROI_masks: 
            print("No ROI masks were defined, please run get_ROI_masks first")
            return None

        # Reset cap
        cap_set_frame(self.video, 0)

        # Perpare arrays to hold the data
        traces = {k:np.zeros(self.video_params[0]) for k in range(self.n_rois)}

        # Extract
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
        raw_traces = trace.copy()

        # Split blue and violet traces
        traces = split_blue_violet_channels(traces)

        # Remove double exponential
        traces = self.remove_double_exponential(traces)

        # Compute DF/F
        traces = self.compute_dff(traces)

        # Regress violet from blue. 
        traces = self.regress_violet_from_blue(traces)
        
        # Save and return
        print("Extraction completed.")
        print("Saving  raw traces at: {}".format(self.save_raw_path))
        raw_traces.to_hdf(self.savsave_raw_pathe_path, key='hdf')
        print("Saving  processed traces at: {}".format(self.save_path))
        traces.to_hdf(self.savsave_pathe_path, key='hdf')
        return raw_traces, traces

    # ---------------------------------------------------------------------------- #
    #                                EXTRACTING DF/F                               #
    # ---------------------------------------------------------------------------- #
        
    def double_exponential(x, a, b, c, d):
        return a * np.exp(b * x) + c * np.exp(d*x)

    def remove_exponential_from_trace(x, y):
        """ Fits a double exponential to the data and returns the results
        
            :param x: np.array with time indices
            :param y: np.array with signal

            :returns: np.array with doble exponential corrected out
        """
        popt, pcov = curve_fit(double_exponential, x, y, 
                            maxfev=2000, 
                            p0=(1.0,  -1e-6, 1.0,  -1e-6),
                            bounds = [[1, -1e-1, 1, -1e-1], [100, 0, 100, 0]])

        fitted_doubleexp = double_exponential(x, *popt)
        y_pred = y - (fitted_doubleexp - np.min(fitted_doubleexp))
        return y_pred

    def remove_double_exponential(self, traces):
        time = np.arange(len(traces))
        for column in traces.columns:
            tracees[column] = remove_exponential_from_trace(time, traces.column.values)

        return traces

    def compute_dff(self, traces):
        for column in traces.columns:
            trace = traces[column].values
            baseline = np.nanmedian(trace)
            traces[column] = (trace-baseline)/baseline
        return traces

    def regress_violet_from_blue(self, traces):
        for roi_n in np.arange(self.n_rois):
            blue = traces[str(roi_n)+'_blue'].values
            violet = traces[str(roi_n)+'_violet'].values

            regressor = LinearRegression()  
            regressor.fit(violet.reshape(-1, 1), blue.reshape(-1, 1))
            expected_blue = regressor.predict(violet)        
            corrected_blue = (blue - expected_blue)/expected_blue
            traces[str(roi_n)+'_dff'] = corrected_blue
        return traces


if __name__ == "__main__":
    video = "Z:\\swc\\branco\\rig_photometry\\tests\\200205\\200205_mantis_test_longer_exposure_noaudio\\FP_calcium_and_behaviour(0)-FP_calcium_camera.mp4"


    se = SignalExtraction(video, overwrite=True)
    se.get_ROI_masks()
    se.extract_signal()




