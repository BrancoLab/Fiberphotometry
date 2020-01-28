import sys
sys.path.append("./")

import os 
import pandas as pd
import numpy as np
from functools import partial
from moviepy.editor import VideoClip
import cv2

if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

from analysis_utils import setup
from analysis_utils import blueled, blue_dff_color, violetled, motion_color, ldr_color

from fcutils.maths.filtering import *
from fcutils.maths.stimuli_detection import *
from fcutils.plotting.plotting_utils import set_figure_subplots_aspect




# ? define folder to process and fps
# folder = '/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/fiberphot_data/200124_fiberp/240120_id_994382_freelymoving_twofibers_3'
folder=r'K:\191213_994832_right_3'
fps = 14 # fps at which we acquired the experiment's videos


# ---------------------------------------------------------------------------- #
#                                 FRAME CREATOR                                #
# ---------------------------------------------------------------------------- #
def make_frame(data, n_fibers, videos, fps, start_frame, t):
    """ 
        This function is called by make_video below to create each frame in the video.
        The frame is crated as a matplotlib figure and then added to the video.

        :param data: pd.dataframe with sensors data
        :param n_fibers: number of channels in recording
        :param videos: dictionary with opencv caps 
        :param fps: int
        :param t: float, time in the video, used to get framenumber
    """
    # ---------------------------------------------------------------------------- #
    def add_videoframe(ax, cavid, behavid, framen, rescale_fact):
        """ 
            Adds a videoframe to a subplot in the frame. 

            :param ax: matplotlib Axes object
            :param video: opencv cap
            :param framen: int number of frame in video
            :param rescale_fact: float, positive numbers reduce frame size and negative enlarge it
        """
        # Get frame from video
        frames = []
        for video in [cavid, behavid]:
            video.set(1,framen)
            ret, frame = video.read()
            if not ret:
                raise ValueError("Could not read frame {} from video".format(framen))
            
            # scale frame
            if rescale_fact>0: # make it smaller
                frame = frame[::rescale_fact,::rescale_fact]

            frames.append(frame)

        # Scale down the calcium frame and insert it into main frame
        caframe = frames[0][::2,::2, 0]
        caframe = np.pad(caframe, 10, constant_values=255) # Add a white border 
        caframe = np.repeat(caframe[:, :, np.newaxis], 3, axis=2)

        frame = frames[1]
        frame[:caframe.shape[0], :caframe.shape[1]] = caframe

        # Add to ax
        ax.imshow(frame, interpolation="nearest", cmap='gray', aspect='equal')
        ax.axis('off')

    def add_signal_plot(ax, data, framen, nframes, **kwargs):
        """
            Adds a line plot to one of the subplots.

            :param ax: matplotlib Axes object
            :param data: np.array with the whole data for one channel in the recording
            :param framen: int, frame number
            :param nframes: int, number of frames before and after framen to show from the data
            :param **kwargs: to style the line plot
        """
        # Pad data
        pad_size = 200
        framen += pad_size
        y = np.pad(data, (pad_size, pad_size))

        # Plot past frames
        ypre = y[framen-nframes:framen]
        xpre = np.arange(len(ypre))
        ax.plot(xpre, ypre, alpha=.6, **kwargs)

        # Plot future frames
        ypost = y[framen:framen+nframes]
        xpost = np.arange(len(ypost))+len(xpre)
        ax.plot(xpost, ypost, **kwargs)

        # Plot
        ax.axvline(nframes, ls='--', lw=2, color='k', alpha=.6)

        # Set x axis
        x = np.linspace(0, nframes*2, 5).astype(np.int16)
        xl = [framen-nframes, int(framen-nframes*.5), framen, 
                int(framen+nframes*.5), framen+nframes]
        ax.set(xticks=x, xticklabels=xl)

    # ---------------------------------------------------------------------------- #
    # Create subplots
    f, axes = plt.subplots(3, 2, figsize=(20, 20),\
                gridspec_kw={'width_ratios': [2, 3],
                            'height_ratios':[3, 1, 1]})
    fibers_frame_ax = axes[0][0]
    behav_frame_ax = axes[0][1]
    motion_ax = axes[1][1]
    ldr_ax = axes[2][1]
    blue_ax = axes[1][0]
    violet_ax = axes[2][0]

    # get frame number
    framen = int(t*fps)+start_frame

    # add video frames to image
    add_videoframe(fibers_frame_ax, videos['calcium'],  videos['behaviour'], framen, 2)

    # Add signals to image
    n_frames_in_plot = 60  
    add_signal_plot(motion_ax, data['behav_mvmt'], framen, n_frames_in_plot, color=motion_color, lw=3)
    add_signal_plot(ldr_ax, data['ldr'], framen, n_frames_in_plot, color=ldr_color, lw=3)
    add_signal_plot(blue_ax, data['ch_0_signal'], framen, n_frames_in_plot, color=blueled, lw=3)
    add_signal_plot(violet_ax, data['ch_0_motion'], framen, n_frames_in_plot, color=violetled, lw=3)

    # Decorate axes
    fibers_frame_ax.set(xticks=[], yticks=[])
    behav_frame_ax.set(xticks=[], yticks=[])
    motion_ax.set(title='Behav. frame. motion', ylabel='movement (a.u.)', xlabel='frame')
    ldr_ax.set(title='LDR signal', ylabel='signal (V)', xlabel='frame')
    blue_ax.set(title='Blue led', ylabel='intensity', xlabel='frame')
    violet_ax.set(title='Violet led', ylabel='intensity', xlabel='frame')

    # improve aestethics
    # Use these values to change subplots aspect
    sns.despine(offset=10, trim=True)
    set_figure_subplots_aspect(
        left  = 0.125,  # the left side of the subplots of the figure
        right = 0.9,    # the right side of the subplots of the figure
        bottom = 0.06,   # the bottom of the subplots of the figure
        top = 0.96,      # the top of the subplots of the figure
        wspace = 0.2,   # the amount of width reserved for blank space between subplots
        hspace = 0.3,   # the amount of height reserved for white space between subplots
    )

    # export image
    f.tight_layout()
    f.canvas.draw()
    img = np.array(f.canvas.renderer.buffer_rgba())
    plt.close(f)

    return img[:, :, :3]


# ---------------------------------------------------------------------------- #
#                                 VIDEO CREATOR                                #
# ---------------------------------------------------------------------------- #
def make_video(folder, fps, overwrite=False, padding=60, start_frame=0, end_frame=-1, **kwargs):
    """
        Creates a 'composite' video with behaviour and calcium data

        :param folder: str, path to a folder with video and sensors data
        :param fps: int, fps of experiment
        :param overwrite: bool if False it avoids overwriting a previously generated video
    """
    # setup
    files, outpath, data, n_fibers = setup(folder,  "_comp_{}_{}.mp4".format(start_frame, end_frame),\
         overwrite, **kwargs)
    if files is None: 
        return
    if len(data) < 2000: 
        return # Ignore recordings that are too short

    # Open videos as caps
    caps = {k: cv2.VideoCapture(f) for k,f in files.items() 
            if '.mp4' in f or '.avi' in f and f is not None}

    # Create video
    cut_data = data[start_frame:end_frame]
    duration = len(cut_data)/fps
    animation = VideoClip(partial(make_frame, data, n_fibers, caps, fps, start_frame), 
                        duration=duration,  has_constant_size=True)

    # Export/save
    animation.write_videofile(outpath, fps=fps) 

# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    make_video(folder, fps, overwrite=True, start_frame=100, end_frame=140)