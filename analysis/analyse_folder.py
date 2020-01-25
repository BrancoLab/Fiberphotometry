"""
    Scripts to automate analysis of all experiments in a folder
"""
import sys
sys.path.append('./')

import os


from analysis.videomaker import make_video
from utils.file_io import get_subdirs

folder = "/Volumes/swc/branco/rig_photometry/tests"

make_summary_plots = True
make_composite_video = True
overwrite = False

for si, subdir in enumerate(get_subdirs(folder)):
    # get the date at which the experiment was executed to adjust some params
    print("Processing subdirectory: {} - #{}".format(
            os.path.split(subdir)[-1], si+1))
    if "_" in subdir:
        date = int(os.path.split(subdir)[-1].split("_")[0])
    else:
        date = int(subdir)
    if date < 200000: 
        invert=True, 
        fps=14
    else:
        invert=False
        fps=10

    # Iterate over the sub subdirs
    for ssi, subsub in enumerate(get_subdirs(subdir)):
        print("     processing sub subdir: {} - #{}".format(
            os.path.split(subsub)[-1]))

        if make_composite_video:
            print("         making composite video")
            make_video(subsub, fps, invert=invert, overwrite=overwrite)




