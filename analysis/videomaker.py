import sys
sys.path.append("./")

import os 
import pandas as pd
import numpy as np
from functools import partial
from moviepy.editor import VideoClip
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import seaborn as sns # used to style plots

from utils.maths.filtering import *
from utils.maths.stimuli_detection import *
from utils.colors import *
from analysis_utils import get_files_in_folder, get_data_from_sensors_csv

# ? Define a bunch of colors
blueled = lightskyblue
violetled = violet
motion_color = thistle
ldr_color = salmon


def make_frame(data, n_fibers, videos, fps, t):
    def add_videoframe(ax, video, framen, rescale_fact):
        video.set(1,framen)
        ret, frame = video.read()
        if not ret:
            raise ValueError("Could not read frame {} from video".format(framen))
        ax.imshow(frame[::rescale_fact,::rescale_fact], interpolation="nearest", cmap='gray')

    def add_signal_plot(ax, data, framen, nframes, **kwargs):
        if framen < nframes:
            toplot = data[:framen]
        elif framen + nframes > len(data):
            toplot = data[framen:]
            toplot = np.concatenate([toplot, np.zeros(2*nframes)])
        else:
            toplot = data[framen-nframes : framen+nframes]
        
        ax.plot(toplot, **kwargs)
        ax.axvline(nframes, ls='--', lw=2, color='k', alpha=.6)

        x = np.linspace(0, nframes, 5).astype(np.int16)
        xl = [framen-nframes, int(framen-nframes*.5), framen, 
                int(framen+nframes*.5), framen+nframes]
        ax.set(xticks=x, yticks=xl)


    """ returns an image of the frame at time t """
    # Create subplots
    f, axes = plt.subplots(4, 2, figsize=(20, 16),\
                gridspec_kw={'width_ratios': [2, 2],
                            'height_ratios':[2, 1, 1, 2]})
    fibers_frame_ax = axes[0][0]
    behav_frame_ax = axes[0][1]
    motion_ax = axes[1][1]
    ldr_ax = axes[2][1]

    # get frame number
    framen = int(t*fps)

    # add frames to image
    add_videoframe(fibers_frame_ax, videos['calcium'], framen, 2)
    add_videoframe(behav_frame_ax, videos['behaviour'], framen, 1)

    # Add signals to image
    add_signal_plot(motion_ax, data['behav_mvmt'], framen, 20, color=motion_color, lw=3)
    add_signal_plot(ldr_ax, data['ldr'], framen, 20, color=ldr_color, lw=3)

    # Decorate axes
    fibers_frame_ax.set(xticks=[], yticks=[])
    behav_frame_ax.set(xticks=[], yticks=[])
    motion_ax.set(title='Behav. frame. motion', ylabel='movement (a.u.)', xlabel='frame')
    ldr_ax.set(title='LDR signal', ylabel='signal (V)', xlabel='frame')

    # export image
    f.canvas.draw()
    img = np.array(f.canvas.renderer.buffer_rgba())
    plt.close(f)

    return img[:, :, :3]


def make_video(folder, fps):
    # Get files
    files = get_files_in_folder(folder)

    # Open videos as caps
    caps = {k: cv2.VideoCapture(f) for k,f in files.items() if '.mp4' in f or '.avi' in f}

    # Get sensors data
    data, n_fibers = get_data_from_sensors_csv(files['sensors'])
    data = data [:60]

    # Create video
    duration = len(data)/fps
    animation = VideoClip(partial(make_frame, data, n_fibers, caps, fps), 
                        duration=duration)

    # Export/save
    name = os.path.split(folder)[-1]
    animation.write_videofile("{}.mp4".format(os.path.join(folder, name)), fps=fps) # export as video


if __name__ == "__main__":
    folder = '/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/fiberphot_data/200124_fiberp/240120_id_994382_freelymoving_twofibers_3'
    make_video(folder, 10)