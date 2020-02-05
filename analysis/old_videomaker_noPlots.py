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
folder=r"F:\240120_id_994382_freelymoving_twofibers_3"
fps =15 # fps at which we acquired the experiment's videos

def scalebar(ax, hideTicks=True, hideFrame=True, fontSize=8, scaleXsize=None, scaleYsize=None, scaleXunits="", scaleYunits="", lineWidth=2): 
     """ 
     Add an L-shaped scalebar to the current figure. 
     This removes current axis labels, ticks, and the figure frame. 
     """ 
  
     # calculate the current data area 
     x1, x2 = ax.get_xlim()
     y1, y2 = ax.get_ylim()  # bounds 
     xc, yc = (x1+x2)/2, (y1+y2)/2  # center point 
     xs, ys = abs(x2-x1), abs(y2-y1)  # span 
  
     # determine how big we want the scalebar to be 
     if scaleXsize == None: 
         scaleXsize = abs(ax.get_xticks()[1]-ax.get_xticks()[0])/2 
     if scaleYsize ==None: 
         scaleYsize = abs(ax.get_yticks()[1]-ax.get_yticks()[0])/2 
  
     # create the scale bar labels 
     lblX = str(scaleXsize) 
     lblY = str(scaleYsize) 
  
     # prevent units unecessarially ending in ".0" 
     if lblX.endswith(".0"): 
         lblX = lblX[:-2] 
     if lblY.endswith(".0"): 
         lblY = lblY[:-2] 
  
     if scaleXunits == "sec" and "." in lblX: 
         lblX = str(int(float(lblX)*1000)) 
         scaleXunits = "ms" 
  
     # add units to the labels 
     lblX = lblX+" "+scaleXunits 
     lblY = lblY+" "+scaleYunits 
     lblX = lblX.strip() 
     lblY = lblY.strip() 
  
     # determine the dimensions of the scalebar 
     scaleBarPadX = 0.10 
     scaleBarPadY = 0.05 
     scaleBarX = x2-scaleBarPadX*xs 
     scaleBarX2 = scaleBarX-scaleXsize 
     scaleBarY = y1+scaleBarPadY*ys 
     scaleBarY2 = scaleBarY+scaleYsize 
  
     # determine the center of the scalebar (where text will go) 
     scaleBarXc = (scaleBarX+scaleBarX2)/2 
     scaleBarYc = (scaleBarY+scaleBarY2)/2 
  
     # create a scalebar point array suitable for plotting as a line 
     scaleBarXs = [scaleBarX2, scaleBarX, scaleBarX] 
     scaleBarYs = [scaleBarY, scaleBarY, scaleBarY2] 
  
     # the text shouldn't touch the scalebar, so calculate how much to pad it 
     lblPadMult = .005 
     lblPadMult += .002*lineWidth 
     lblPadX = xs*lblPadMult 
     lblPadY = ys*lblPadMult 
  
     # hide the old tick marks 
     if hideTicks: 
         ax.get_yaxis().set_visible(False) 
         ax.get_xaxis().set_visible(False) 
  
     # hide the square around the image 
     if hideFrame: 
         ax.spines['top'].set_visible(False) 
         ax.spines['right'].set_visible(False) 
         ax.spines['bottom'].set_visible(False) 
         ax.spines['left'].set_visible(False) 
  
     # now do the plotting 
     ax.plot(scaleBarXs, scaleBarYs, 'k-', lw=lineWidth) 
     ax.text(scaleBarXc, scaleBarY-lblPadY, lblX, 
              ha='center', va='top', fontsize=fontSize) 
     ax.text(scaleBarX+lblPadX, scaleBarYc, lblY, 
              ha='left', va='center', fontsize=fontSize) 

# ---------------------------------------------------------------------------- #
#                                 FRAME CREATOR                                #
# ---------------------------------------------------------------------------- #
def make_frame(data, n_fibers, videos, fps, start_frame, end_frame, xlabel, t):
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
    def add_videoframe_inset(ax, cavid, behavid, framen, rescale_fact):
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
        caframe = np.pad(caframe, 5, constant_values=255) # Add a white border 
        caframe = np.repeat(caframe[:, :, np.newaxis], 3, axis=2)

        frame = frames[1]
        frame[:caframe.shape[0], :caframe.shape[1]] = caframe

        # Add to ax
        ax.imshow(frame, interpolation="nearest", cmap='gray', aspect='equal')
        ax.axis('off')
    
    def add_videoframe(ax, video, framen, rescale_fact):
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
        
        # scale frame
        if rescale_fact>0: # make it smaller
            frame = frame[::rescale_fact,::rescale_fact]
        elif rescale_fact<0: # make it larger
            w = frame.shape[0] * np.abs(rescale_fact)
            h = frame.shape[1] * np.abs(rescale_fact)
            frame = cv2.resize(frame, (h, w))

        # Add to ax
        ax.imshow(frame, interpolation="nearest", cmap='gray', aspect='equal')
        ax.axis('off')
    
    def add_signal_plot(ax, data, framen, n_frames_b4, n_frames_aft, start_frame, end_frame, xlabel = 'frame', **kwargs):
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
        y = np.pad(data, (pad_size, pad_size), constant_values = np.nan)

        # Plot past frames
        ypre = y[framen-n_frames_b4:framen+1]
        xpre = np.arange(len(ypre))
        ax.plot(xpre, ypre, alpha=1, **kwargs)

        # Plot future frames
        ypost = y[framen:framen+n_frames_aft]
        xpost = np.arange(len(ypost))+len(xpre)-1
        ax.plot(xpost, ypost, alpha = 0.5, **kwargs)

        # Plot
        ax.axvline(n_frames_b4, ls='--', lw=2, color='k', alpha=.6)

        # Set x axis
        x = np.linspace(0, n_frames_b4+n_frames_aft, 5).astype(np.int16)
        xl = [framen-n_frames_b4, int(framen-n_frames_b4*.5), framen, 
                int(framen+n_frames_aft*.5), framen+n_frames_aft]
        if 'time' in xlabel:
            xl = [int(x / fps) for x in xl]
        ax.set(xticks=x, xticklabels=xl)
        
#        ax.set_ylim(np.nanmin(data[np.min([2,start_frame-nframes]):np.max([len(data),end_frame+nframes])]),
#                    np.nanmax(data[np.min([2,start_frame-nframes]):np.max([len(data),end_frame+nframes])]))

    # ---------------------------------------------------------------------------- #
    # Create subplots
    
    plt.ioff()
    
    f, axes = plt.subplots(1, 2, figsize=(20, 10),\
                gridspec_kw={'width_ratios': [1, 3]})
    fibers_frame_ax = axes[0]
    behav_frame_ax = axes[1]

    
    # get frame number
    framen= int(t*fps)+start_frame 

    # add video frames to image
  #  add_videoframe_inset(videos_frame_ax, videos['calcium'],  videos['behaviour'], framen, 2)
    add_videoframe(fibers_frame_ax, videos['calcium'], framen, 2)
    add_videoframe(behav_frame_ax, videos['behaviour'], framen, -2)
    
      
    # Decorate axes
    fibers_frame_ax.set(xticks=[], yticks=[])
    behav_frame_ax.set(xticks=[], yticks=[])

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
    f.tight_layout(rect=(0,0,0.95,1))
    f.canvas.draw()
    img = np.array(f.canvas.renderer.buffer_rgba())
    plt.close(f)
    
    return img[:, :, :3]


# ---------------------------------------------------------------------------- #
#                                 VIDEO CREATOR                                #
# ---------------------------------------------------------------------------- #
def make_video(folder, fps, overwrite=False, padding=60, start_frame=2, end_frame=-1, xlabel = 'frame', **kwargs):
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
    animation = VideoClip(partial(make_frame, data, n_fibers, caps, fps, start_frame,end_frame, xlabel), 
                        duration=duration,  has_constant_size=True)

    # Export/save
    animation.write_videofile(outpath, fps=fps,logger=None) 

# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    make_video(folder, fps, overwrite=True, start_frame=3605, end_frame=4005, xlabel = 'time (s)')







#----------------------------------------------------------------------------------
