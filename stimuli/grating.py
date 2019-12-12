import psychopy.visual
import psychopy.event
import time
import numpy as np
from tqdm import tqdm


# PARAMS
repeats_per_orientation = 3
orientations = [0, 45, 90]
spatial_freq = 5.0 / 400.0  
n_frames_on = 50
n_frames_off = 25
speed = 10


# CREATE STUFF
white, black, gray = [1, 1, 1], [-1, -1, -1], [1, 1, 1]

win = psychopy.visual.Window(
    pos=[50, 50],
    size=[800, 800],
    units="pix",
    fullscr=False
)

square = psychopy.visual.Rect(
    win=win,
    pos=[-375, 375], 
    width=50,
    height=50, 
    units="pix",
    fillColor=black
)

grating = psychopy.visual.GratingStim(
    win=win,
    ori=orientations[0], 
    units="pix",
    size=[800, 800]
)
grating.sf = spatial_freq

# COMPUTE FRAMES
orientations = np.repeat(orientations, repeats_per_orientation)

square_frames = np.hstack(np.tile(np.array([np.ones(n_frames_on), np.zeros(n_frames_off)]), len(orientations)))
ors = np.hstack([np.ones(n_frames_on+n_frames_off)*o for o in orientations])
phases = np.hstack([np.hstack([np.linspace(0, speed, n_frames_on), np.zeros(n_frames_off)]) for o in orientations])

if phases.shape != ors.shape or phases.shape != square_frames.shape:
    raise ValueError("Something went wrong with frames calculation")
else:
    print("Total duration: {} stimuli over {}".format(len(orientations), phases.shape[0]))

# Animate stimuli
curr_orientation = orientations[0]
for s, o, p in tqdm(zip(square_frames, ors, phases)):
    if s:
        square.color = white
    else:
        square.color = black

    if p==0:
        grating.opacity=0
    else:
        grating.opacity = 1

    if o != curr_orientation:
        grating = psychopy.visual.GratingStim(
            win=win,
            ori=o, 
            units="pix",
            size=[800, 800]
        )
        grating.sf = spatial_freq

        curr_orientation = o
        print("\n\nStarting stim with orientation: {}".format(round(o)))

    grating.phase = p

    grating.draw()
    square.draw()
    win.flip()

win.close()