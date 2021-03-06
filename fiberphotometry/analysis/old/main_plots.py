"""
    Bunch of functions to make standard summary plots for a session
"""
import sys
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')

import seaborn as sns
from scipy import stats
import random
import numpy as np 

from analysis_utils import setup, get_stimuli_from_ldr
from analysis_utils import blueled, blue_dff_color, violetled, motion_color, ldr_color

from fcutils.file_io.utils import check_file_exists
from fcutils.plotting.plotting_utils import *
from fcutils.plotting.colors import desaturate_color
from fcutils.plotting.plot_elements import plot_shaded_withline, ortholines


folder = r"F:\240120_id_994382_freelymoving_twofibers_3"


def plot_session_traces(folder, overwrite=True, **kwargs):
    def plot_trace(ax, x, y, title, xlabel, ylabel, **kwargs):
        plot_shaded_withline(ax, x, y, alpha=.4, **kwargs)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)

    # ---------------------------------------------------------------------------- #
    files, outpath, data, n_fibers = setup(folder, "_session_traces.png", overwrite, **kwargs)
    if data is None: return
    stim_onset, stim_offset = get_stimuli_from_ldr(data['ldr'])

    f, axarr = create_figure(subplots=True, ncols=1, nrows=2+5*n_fibers, figsize=(20, 12), sharex=True)
    
    x = np.arange(len(data.ldr.values)-2)

    # plot ldr and movement traces
    plot_trace(axarr[0], x, data.ldr.values[2:], 'LDR', None, 'v', color=ldr_color, lw=3, z=np.min(data.ldr.values[2:]))
    plot_trace(axarr[1], x, data.behav_mvmt.values[2:], 'Motion', None, 'a.u.', color=motion_color, lw=3, z=np.min(data.behav_mvmt.values[2:]))

    # Plot fiber traces
    for i in range(n_fibers):
        blue = data['ch_{}_signal'.format(i)].values[2:]
        violet = data['ch_{}_motion'.format(i)].values[2:]
        blue_no_bleach = data['ch_{}_blue_nobleach'.format(i)].values[2:]
        violet_no_bleach = data['ch_{}_violet_nobleach'.format(i)].values[2:]
        blue_dff = data['ch_{}_blue_dff'.format(i)].values[2:]
        violet_dff = data['ch_{}_violet_dff'.format(i)].values[2:]
        expected_blue_dff = data['ch_{}_expected_blue_dff'.format(i)].values[2:]
        corrected = data['ch_{}_corrected'.format(i)].values[2:]
        corrected_regOnly = data['ch_{}_corrected_regOnly'.format(i)].values[2:]

        plot_trace(axarr[i+2], x, blue, 
                '$470nm$', None, 'a.u.', color=blueled, lw=3, z=np.min(blue))
        plot_trace(axarr[i+3], x, blue_dff, 
                '470nm after bleaching correction', None, 'dF/F', color=blueled, lw=3, z=0)
        plot_trace(axarr[i+4], x, violet, 
                '$405nm$', None, 'a.u.', color=violetled, lw=3, z=np.min(violet))
        plot_trace(axarr[i+5], x, violet_dff, 
                '405nm after bleaching correction', None, 'dF/F', color=violetled, lw=3, z=0)
        if i == n_fibers-1:
            xlabel = 'Frames'
        else:
            xlabel = None
        plot_trace(axarr[i+6], x, corrected, '470nm after motion correction', 
                                xlabel, 'dF/F.', color=blue_dff_color, lw=3, z=np.nanmin(corrected))
#        plot_trace(axarr[i+6], x, corrected_regOnly, '470nm after motion correction.', 
#                                xlabel, 'dF/F.', color='gray', lw=3, z=np.nanmin(corrected))
    # Plot stim onsets
    for ax in axarr:
        ones = [1 for _ in stim_onset]
        ortholines(ax, ones, list(stim_onset), color=ldr_color, lw=1, ls="--")
        
    # Clean look and save
    sns.despine(offset=2, trim=True)
    set_figure_subplots_aspect(
        left  = 0.125,  # the left side of the subplots of the figure
        right = 0.9,    # the right side of the subplots of the figure
        bottom = 0.06,   # the bottom of the subplots of the figure
        top = 0.96,      # the top of the subplots of the figure
        wspace = 0.4,   # the amount of width reserved for blank space between subplots
        hspace = 0.5,   # the amount of height reserved for white space between subplots
    )
    

    save_figure(f, outpath.split(".png")[0])
    #close_figure(f)


def plot_session_psth(folder, overwrite=True, baseline_frames = 30, plot_shuffled=True,
                show_individual_trials=False, post_frames=30, **kwargs):
    # Set up data
    files, outpath, data, n_fibers = setup(folder, "_session_psth.png", overwrite, **kwargs)
    if data is None: return
    stim_onset, stim_offset = get_stimuli_from_ldr(data['ldr'])
    
    # Some checks
    if len(stim_onset) < 5: return
    if len(stim_onset) != len(stim_offset): 
        if len(stim_onset) > len(stim_offset):
            stim_onset = stim_onset[:len(stim_offset)]
        else:
            raise ValueError

    stim_duration = np.mean(stim_offset-stim_onset)


    # Get aligned trials data
    trials = [[] for i in range(n_fibers)]
    random_trials = [[] for i in range(n_fibers)]
    for onset in stim_onset:
        for i in range(n_fibers):
            trials[i].append(data['ch_{}_corrected'.format(i)].values[onset-baseline_frames:onset+post_frames])

            random_onset = np.random.randint(0, len(data))
            random_trials[i].append(data['ch_{}_corrected'.format(i)].values[random_onset-baseline_frames:random_onset+post_frames])

    if n_fibers>1:
        f, axarr = create_figure(subplots=True, nrows=n_fibers, sharex=True, figsize=(20, 12))
    else:
        f, ax = create_figure(subplots=False, figsize=(20, 12))
        axarr=[ax]

    for i,(ax, trs, random_trs) in enumerate(zip(axarr, trials, random_trials)):
        # Plot single trials
        nframes = baseline_frames+post_frames
        trs = np.array([t for t in trs if len(t) == nframes])
        if show_individual_trials:
            ax.plot(trs.T, color=desaturate_color(blue_dff_color), lw=2, alpha=.5)

        # Plot mean + error for real data
        mn, err = np.nanmean(trs, axis=0), stats.sem(trs, axis=0,nan_policy='omit')
        x = np.arange(len(mn))
        ax.plot(x, mn, lw=4, color=blue_dff_color, alpha=1, zorder=99, label='signal')
        ax.fill_between(x, mn-err, mn+err, color=blue_dff_color, alpha=.3, zorder=90)

        # plot mean + error for shuffled data
        if plot_shuffled:
            rndtrs = np.array([t for t in random_trs if len(t) == nframes])
            mn, err = np.nanmean(rndtrs, axis=0), stats.sem(rndtrs, axis=0,nan_policy='omit')
            ax.plot(x, mn, lw=4, color=[.6, .6, .6], alpha=.8, zorder=95, label='shuffled')
            ax.fill_between(x, mn-err, mn+err, color=[.6, .6, .6], alpha=.1, zorder=80)

        ax.legend()
        ax.axvline(baseline_frames, lw=3, color=[.5, .5, .5], ls="--")
        ax.axvline(baseline_frames+stim_duration, lw=3, color=[.3, .3, .3], ls="--")

        # clean axis
        ax.set(xticks=[0, baseline_frames, baseline_frames+post_frames], xticklabels=[-baseline_frames, 0, post_frames],
                title='Channel {} psth'.format(i), xlabel='frames', ylabel='corrected 470nm')

    # Clean look and save
    sns.despine(offset=2, trim=True)
    set_figure_subplots_aspect(
        left  = 0.125,  # the left side of the subplots of the figure
        right = 0.9,    # the right side of the subplots of the figure
        bottom = 0.06,   # the bottom of the subplots of the figure
        top = 0.96,      # the top of the subplots of the figure
        wspace = 0.4,   # the amount of width reserved for blank space between subplots
        hspace = 0.5,   # the amount of height reserved for white space between subplots
    )
    
    save_figure(f, outpath.split(".png")[0])
    #close_figure(f)

def plot_session_psth_locomotion(folder, overwrite=True, baseline_frames = 30, plot_shuffled=True,
                show_individual_trials=False, post_frames=30, **kwargs):
    # Set up data
    files, outpath, data, n_fibers = setup(folder, "_session_psth_locomotion.png", overwrite, **kwargs)
    if data is None: return
    stim_onset, stim_offset = get_stimuli_from_ldr(data['ldr'])
    
    # Some checks
    if len(stim_onset) < 5: return
    if len(stim_onset) != len(stim_offset): 
        if len(stim_onset) > len(stim_offset):
            stim_onset = stim_onset[:len(stim_offset)]
        else:
            raise ValueError

    stim_duration = np.mean(stim_offset-stim_onset)

    low_locomotion = np.where(data.behav_mvmt.values[2:]<np.percentile(data.behav_mvmt.values[2:],40))[0]
    high_locomotion = np.where(data.behav_mvmt.values[2:]>np.percentile(data.behav_mvmt.values[2:],60))[0]
    
    # Get aligned trials data
    trials = [[] for i in range(n_fibers)]
    low_trials = [[] for i in range(n_fibers)]
    high_trials = [[] for i in range(n_fibers)]
    random_trials = [[] for i in range(n_fibers)]
    for onset in stim_onset:
        for i in range(n_fibers):
            trials[i].append(data['ch_{}_corrected'.format(i)].values[onset-baseline_frames:onset+post_frames])
            
            if onset in low_locomotion: 
                low_trials[i].append(data['ch_{}_corrected'.format(i)].values[onset-baseline_frames:onset+post_frames])
            elif onset in high_locomotion:
                high_trials[i].append(data['ch_{}_corrected'.format(i)].values[onset-baseline_frames:onset+post_frames])

            random_onset = np.random.randint(0, len(data))
            random_trials[i].append(data['ch_{}_corrected'.format(i)].values[random_onset-baseline_frames:random_onset+post_frames])

    if n_fibers>1:
        f, axarr = create_figure(subplots=True, nrows=n_fibers, sharex=True, figsize=(20, 12))
    else:
        f, ax = create_figure(subplots=False, figsize=(20, 12))
        axarr=[ax]

    for i,(ax, trs,low_trs,high_trs, random_trs) in enumerate(zip(axarr, trials, low_trials, high_trials, random_trials)):
        # Plot single trials
        nframes = baseline_frames+post_frames
        trs = np.array([t for t in trs if len(t) == nframes])
        if show_individual_trials:
            ax.plot(trs.T, color=desaturate_color(blue_dff_color), lw=2, alpha=.5)

        # Plot mean + error for real data
        mn, err = np.nanmean(trs, axis=0), stats.sem(trs, axis=0,nan_policy='omit')
        x = np.arange(len(mn))
        ax.plot(x, mn, lw=4, color=blue_dff_color, alpha=1, zorder=99, label='signal')
        ax.fill_between(x, mn-err, mn+err, color=blue_dff_color, alpha=.3, zorder=90)

        mn, err = np.nanmean(low_trs, axis=0), stats.sem(low_trs, axis=0,nan_policy='omit')
        x = np.arange(len(mn))
        ax.plot(x, mn, lw=4, color=ldr_color, alpha=0.5, zorder=99, label='low locomotion')
        ax.fill_between(x, mn-err, mn+err, color=ldr_color, alpha=.2, zorder=90)
        
        mn, err = np.nanmean(high_trs, axis=0), stats.sem(high_trs, axis=0,nan_policy='omit')
        x = np.arange(len(mn))
        ax.plot(x, mn, lw=4, color=(0.2353, 0.7020, 0.4431), alpha=0.5, zorder=99, label='high_locomotion')
        ax.fill_between(x, mn-err, mn+err, color=violetled, alpha=.2, zorder=90)
        
        # plot mean + error for shuffled data
        if plot_shuffled:
            rndtrs = np.array([t for t in random_trs if len(t) == nframes])
            mn, err = np.nanmean(rndtrs, axis=0), stats.sem(rndtrs, axis=0,nan_policy='omit')
            ax.plot(x, mn, lw=4, color=[.6, .6, .6], alpha=.8, zorder=95, label='shuffled')
            ax.fill_between(x, mn-err, mn+err, color=[.6, .6, .6], alpha=.1, zorder=80)

        ax.legend()
        ax.axvline(baseline_frames, lw=3, color=[.5, .5, .5], ls="--")
        ax.axvline(baseline_frames+stim_duration, lw=3, color=[.3, .3, .3], ls="--")

        # clean axis
        ax.set(xticks=[0, baseline_frames, baseline_frames+post_frames], xticklabels=[-baseline_frames, 0, post_frames],
                title='Channel {} psth'.format(i), xlabel='frames', ylabel='corrected 470nm')

    # Clean look and save
    sns.despine(offset=2, trim=True)
    set_figure_subplots_aspect(
        left  = 0.125,  # the left side of the subplots of the figure
        right = 0.9,    # the right side of the subplots of the figure
        bottom = 0.06,   # the bottom of the subplots of the figure
        top = 0.96,      # the top of the subplots of the figure
        wspace = 0.4,   # the amount of width reserved for blank space between subplots
        hspace = 0.5,   # the amount of height reserved for white space between subplots
    )
    
    save_figure(f, outpath.split(".png")[0])
    
if __name__ == "__main__":
    plot_session_traces(folder, overwrite=True)