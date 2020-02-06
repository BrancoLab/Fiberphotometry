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
folder=r"F:\240120_id_994383_freelymoving2"
fps = 14 # fps at which we acquired the experiment's videos

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
        caframe = np.pad(caframe, 5, constant_values=255) # Add a white border 
        caframe = np.repeat(caframe[:, :, np.newaxis], 3, axis=2)

        frame = frames[1]
        frame[:caframe.shape[0], :caframe.shape[1]] = caframe

        # Add to ax
        ax.imshow(frame, interpolation="nearest", cmap='gray', aspect='equal')
        ax.axis('off')
       

    def add_signal_plot(ax, data, framen, n_frames_b4, n_frames_aft, start_frame, end_frame, xlabel = 'frame', legend = [], **kwargs):
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
        if legend:
            ax.plot(xpre, ypre, alpha=1, label = legend, **kwargs)
        else:
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
        
        if legend:
            leg = ax.legend(prop = {'weight':'bold', 'size':17}, loc = 1, frameon = False)
            for line, text in zip(leg.get_lines(), leg.get_texts()):
                text.set_color(line.get_color())
        

    # ---------------------------------------------------------------------------- #
    # Create subplots
    
    plt.ioff()
    
#    for long rectangular videos:
    f = plt.figure(constrained_layout=True,figsize=(28, 10))
    gs = f.add_gridspec(nrows=4, ncols=2, height_ratios=[1,1,1,1],width_ratios=[4,3])
    
#    #for square videos
#    f = plt.figure(constrained_layout=True,figsize=(20, 10))
#    gs = f.add_gridspec(nrows=4, ncols=2, height_ratios=[1,1,1,1],width_ratios=[2,3]) 
    
    videos_frame_ax = f.add_subplot(gs[:, 0])
    blue_ax = f.add_subplot(gs[0, 1])
    violet_ax = f.add_subplot(gs[1, 1])
    dff_ax = f.add_subplot(gs[2, 1])
    motion_ax = f.add_subplot(gs[3, 1])
    #ldr_ax = motion_ax.twinx()
    #ldr_ax.spines["right"].set_visible(True)
    
    # get frame number
    framen= int(t*fps)+start_frame 

    # add video frames to image
    add_videoframe(videos_frame_ax, videos['calcium'],  videos['behaviour'], framen, 2)

    # Add signals to image
    n_frames_b4 = 90
    n_frames_aft = 60
    
    plot_start = np.max([2,start_frame-n_frames_b4])
    plot_end = np.min([len(data),end_frame+n_frames_aft])
    
    behav_mvmt_norm = (data['behav_mvmt']-np.nanmin(data['behav_mvmt'][plot_start:plot_end]))/(np.nanmax(data['behav_mvmt'][plot_start:plot_end])-np.nanmin(data['behav_mvmt'][plot_start:plot_end]))
    ldr_norm = (data['ldr']-np.nanmin(data['ldr']))/np.nanmax([4.5,np.nanmax(data['ldr'])])
    add_signal_plot(motion_ax, behav_mvmt_norm, framen, n_frames_b4, n_frames_aft, start_frame, end_frame, legend = 'Locomotion', xlabel = xlabel, color=motion_color, lw=3)
    add_signal_plot(motion_ax, ldr_norm, framen,  n_frames_b4, n_frames_aft,  start_frame, end_frame, legend ='Visual stim', xlabel = xlabel, color=ldr_color, lw=3)
    
    add_signal_plot(blue_ax, data['ch_0_signal'], framen,  n_frames_b4, n_frames_aft, start_frame, end_frame, legend = 'Blue LED', xlabel = xlabel, color=blueled, lw=3)
    add_signal_plot(violet_ax, data['ch_0_motion'], framen,  n_frames_b4, n_frames_aft, start_frame, end_frame, legend = 'Violet LED', xlabel = xlabel,  color=violetled, lw=3)
    add_signal_plot(dff_ax, data['ch_0_corrected'], framen,  n_frames_b4, n_frames_aft,  start_frame, end_frame, legend = 'Corrected signal', xlabel = xlabel,  color=blue_dff_color, lw=3)
    
    #set y axis range
    [blue_min, violet_min, dff_min] = [ np.nanmin(data['ch_0_signal'][plot_start:plot_end]),
                                        np.nanmin(data['ch_0_motion'][plot_start:plot_end]),
                                        np.nanmin(data['ch_0_corrected'][plot_start:plot_end])  ]
    
    [blue_max, violet_max, dff_max] = [ np.nanmax(data['ch_0_signal'][plot_start:plot_end]),
                                        np.nanmax(data['ch_0_motion'][plot_start:plot_end]),
                                        np.nanmax(data['ch_0_corrected'][plot_start:plot_end])  ]
    
    raw_halfrange = np.max([blue_max - blue_min, violet_max - violet_min])*1.2 /2
    dff_halfrange = (dff_max - dff_min)*1.2 /2
    
    blue_ax.set_ylim(np.mean([blue_max, blue_min])-raw_halfrange, np.mean([blue_max, blue_min])+raw_halfrange)
    violet_ax.set_ylim(np.mean([violet_max, violet_min])-raw_halfrange, np.mean([violet_max, violet_min])+raw_halfrange)
    dff_ax.set_ylim(np.mean([dff_max, dff_min])-dff_halfrange, np.mean([dff_max, dff_min])+dff_halfrange)
    motion_ax.set_ylim(0,1)
    
    # Decorate axes
    videos_frame_ax.set(xticks=[], yticks=[])
    motion_ax.set(xlabel = 'time (s)')
    
    for [ax, title, color, ylabel] in zip(  [blue_ax, violet_ax, dff_ax,motion_ax],
                                            ['Blue led', 'Violet led', 'Corrected signal', ''], 
                                            [blueled, violetled, blue_dff_color, 'black'],
                                            ['intensity', 'intensity', 'dF/F', 'au']        ):
        
#        ax.set_title(label = title, color = color, fontweight = 'bold')
        ax.set_ylabel(ylabel)
        ax.tick_params(labelsize=15)
        
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(17)
    
#    motion_ax.set_ylabel('Motion (a.u.)', color=motion_color)  
#    ldr_ax.set_ylabel('LDR (a.u.)', color=ldr_color)  
#    ldr_ax.set_ylim(0,1)

    # improve aestethics
    # Use these values to change subplots aspect
    sns.despine(offset=10, trim=True)
#    sns.despine(ax=motion_ax, right=False, left=False) 
 #   sns.despine(ax=ldr_ax, left=True, right=False) 

    set_figure_subplots_aspect(
        left  = 0.03,  # the left side of the subplots of the figure
        right = 0.99,    # the right side of the subplots of the figure
        bottom = 0.05,   # the bottom of the subplots of the figure
        top = 0.95,      # the top of the subplots of the figure
        wspace = 0.15,   # the amount of width reserved for blank space between subplots
        hspace = 0.4,   # the amount of height reserved for white space between subplots
    )

    # export image
#    f.tight_layout(rect=(0,0,0.95,1))
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
    make_video(folder, fps, overwrite=True, start_frame=1740, end_frame=2300, xlabel = 'time (s)')







#----------------------------------------------------------------------------------
