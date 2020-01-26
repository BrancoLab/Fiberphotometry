"""
    Scripts to automate analysis of all experiments in a folder
"""
import sys
sys.path.append('./')

import os


from analysis.videomaker import make_video
from fcutils.file_io.utils import get_subdirs
from analysis.main_plots import plot_session_traces, plot_session_psth

folder = "/Volumes/swc/branco/rig_photometry/tests"

make_summary_plots = True
make_composite_video = True
overwrite = False

for si, subdir in enumerate(get_subdirs(folder)):
    # get the date at which the experiment was executed to adjust some params
    print("Processing subdirectory: {}".format(subdir))
    if "_" in subdir:
        date = int(os.path.split(subdir)[-1].split("_")[0])
    else:
        date = int(subdir)
    if date < 200000: 
        invert=True, 
        fps=14
    else:
        invert=False
        fps=14

    # Iterate over the sub subdirs
    for ssi, subsub in enumerate(get_subdirs(subdir)):
        print("     processing sub subdir: {}".format(subsub))

        if make_composite_video:
            print("         making composite video")
            make_video(subsub, fps, invert=invert, overwrite=overwrite)

        if make_summary_plots:
            print("         making summary plots")
            plot_session_traces(subsub, overwrite=overwrite)
            plot_session_psth(subsub, overwrite=overwrite)





