import psychopy.visual
import psychopy.event
import time
import numpy as np
from tqdm import tqdm

def run():
    # PARAMS
    repeats_per_orientation = 30
    orientations = [0, 45, 90]
    spatial_freq = 5.0 / 400.0  
    n_frames_on = 75
    n_frames_off = 400
    speed = 5


    # CREATE STUFF
    white, black, gray = [1, 1, 1], [-1, -1, -1], [1, 1, 1]

    win = psychopy.visual.Window(
        pos=[50, 50],
        size=[1920, 1080],
        units="pix",
        fullscr=True,
        screen=1
    )

    square = psychopy.visual.Rect(
        win=win,
        pos=[-835, 415],
        width=250,
        height=250, 
        units="pix",
        fillColor=black
    )

    grating_params = dict(
        win=win,
        units="pix",
        size=[8000, 8000]
    )


    grating = psychopy.visual.GratingStim(
        ori=orientations[0], 
        **grating_params
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
                ori=o,
            **grating_params
            )
            grating.sf = spatial_freq

            curr_orientation = o
            print("\n\nStarting stim with orientation: {}".format(round(o)))

        grating.phase = p

        grating.draw()
        square.draw()
        win.flip()

    win.close()

if __name__ == "__main__":
    run()