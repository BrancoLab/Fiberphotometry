"""
    Scripts to automate analysis of all experiments in a folder
"""
import sys
sys.path.append('./')

import os
import multiprocessing as mp

from analysis.videomaker import make_video
from analysis.main_plots import plot_session_traces, plot_session_psth

from fcutils.file_io.utils import get_subdirs
from fcutils.objects import flatten_list


def process_subdir(subsubdir):
    # get the date at which the experiment was executed to adjust some params
    print("Processing subdirectory: {}".format(subsubdir))
    if "_" in subsubdir:
        date = int(os.path.split(subsubdir)[-1].split("_")[0])
    else:
        date = int(subsubdir)
    if date < 200000: 
        invert=True, 
        fps=14
    else:
        invert=False
        fps=14

    if make_composite_video:
        make_video(subsubdir, fps, invert=invert, overwrite=overwrite)

    if make_summary_plots:
        plot_session_traces(subsubdir, overwrite=overwrite)
        plot_session_psth(subsubdir, overwrite=overwrite, plot_shuffled=True, 
                            post_frames=120, baseline_frames=100)


# ---------------------------------------------------------------------------- #
#                              PARALLEL PROCESSING                             #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    # folder = "/nfs/winstor/branco/rig_photometry/tests"
    folder = '/Volumes/swc/branco/rig_photometry/tests'

    make_summary_plots = True
    make_composite_video = False
    overwrite = True

    subdirs = get_subdirs(folder)
    subsub = flatten_list([get_subdirs(sub) for sub in subdirs])
    print("Number of subsubdirs: ", len(subsub))
    print("Number of processors: ", mp.cpu_count())

    pool = mp.Pool(mp.cpu_count())
    results = pool.map(process_subdir, subsub)
    pool.close()





