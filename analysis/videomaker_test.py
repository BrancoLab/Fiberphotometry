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
folder = '/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/fiberphot_data/200124_fiberp/240120_id_994382_freelymoving_twofibers_3'
fps = 10 # fps at which we acquired the experiment's videos


# ---------------------------------------------------------------------------- #
#                                 FRAME CREATOR                                #
# ---------------------------------------------------------------------------- #
   def add_videoframe(ax, video, framen, rescale_fact, update=False):
        """ 
            Adds a videoframe to a subplot in the frame. 

            :param ax: matplotlib Axes object
            :param video: opencv cap
            :param framen: int number of frame in video
            :param rescale_fact: float, positive numbers reduce frame size and negative enlarge it
        """
        # Get frame from video
        video.set(1,framen)
        ret, frame = video.read()
        if not ret:
            raise ValueError("Could not read frame {} from video".format(framen))
        
        # # scale frame
        # if rescale_fact>0: # make it smaller
        #     frame = frame[::rescale_fact,::rescale_fact]
        # elif rescale_fact<0: # make it larger
        #     w = frame.shape[0] * np.abs(rescale_fact)
        #     h = frame.shape[1] * np.abs(rescale_fact)
        #     frame = cv2.resize(frame, (h, w))

        # Add to ax
        if not update:
            ax.imshow(frame, interpolation="nearest", cmap='gray', aspect='equal')
            ax.axis('off')
        else:
            ax.set_data(frame)


def make_frame(figure, axes, t):
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
    fibers_frame_ax, behav_frame_ax, motion_ax, ldr_ax, blue_ax, violet_ax = axes

    # get frame number
    framen = int(t*fps)

    # add video frames to image
    add_videoframe(fibers_frame_ax, videos['calcium'], framen, 2, update=True)
    add_videoframe(behav_frame_ax, videos['behaviour'], framen, -2, update=True)

    # Update axes limits
    for ax in axes:
        ax.set_xlim([framen-60, framen+60])

    # export image
    f.canvas.draw()
    img = np.array(f.canvas.renderer.buffer_rgba())

    return img[:, :, :3]


def initialise_figure(data, n_fibers, caps, fps):
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

    # Add video frames
    add_videoframe(fibers_frame_ax, videos['calcium'], 1, 2)
    add_videoframe(behav_frame_ax, videos['behaviour'], 1, -2)
 
    # Add line plots
    motion_ax.plot(data['behav_mvmt'], color=motion_color, lw=3)
    ldr_ax.plot(data['ldr'], color=ldr_color, lw=3)
    blue_ax.plot(data['ch_0_signal'], color=blueled, lw=3)
    violet_ax.plot(data['ch_0_motion'], color=violetled, lw=3)

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
    f.tight_layout()

    # start interactive mode
    plt.ion()

    return f, [fibers_frame_ax, behav_frame_ax, motion_ax, ldr_ax, blue_ax, violet_ax]

# ---------------------------------------------------------------------------- #
#                                 VIDEO CREATOR                                #
# ---------------------------------------------------------------------------- #
def make_video(folder, fps, overwrite=False, **kwargs):
    """
        Creates a 'composite' video with behaviour and calcium data

        :param folder: str, path to a folder with video and sensors data
        :param fps: int, fps of experiment
        :param overwrite: bool if False it avoids overwriting a previously generated video
    """
    # setup
    files, outpath, data, n_fibers = setup(folder, "_comp.mp4", overwrite, **kwargs)
    if files is None: 
        return
    if len(data) < 2000: 
        return # Ignore recordings that are too short

    # Open videos as caps
    caps = {k: cv2.VideoCapture(f) for k,f in files.items() 
            if '.mp4' in f or '.avi' in f and f is not None}

    # Create video
    duration = len(data)/fps
    f, axes = initialise_figure(data, n_fibers, caps, fps)
    animation = VideoClip(partial(make_frame, f, axes), 
                        duration=duration)

    # Export/save
    animation.write_videofile(outpath, fps=fps) 

# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    make_video(folder, fps, overwrite=True)