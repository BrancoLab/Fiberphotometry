import matplotlib.pyplot as plt

from fcutils.plotting.utils import create_figure

from fiberphotometry.variables import channels_colors

def plot_frame_times(calcium_triggers, blue_triggers, violet_triggers, 
                frame_times, max_x=10000):
    f, axarr =  create_figure(subplots=True, nrows=3, sharex=True)

    for ax, chname, ch, col in zip(axarr, ['calcium_camera', 'blue_led', 'violet_led'], \
                        [calcium_triggers, blue_triggers, violet_triggers], channels_colors):
        ax.plot(ch[:max_x], color=col)

        times = [x for x in frame_times[chname] if x < max_x]
        y = [5 for _ in times]
        ax.scatter(times, y, color=col, s=50, edgecolor='k', zorder=99)

        ax.set(title=f"Frames for {chname}")

    plt.show()
