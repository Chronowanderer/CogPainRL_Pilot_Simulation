import base64
import IPython
import imageio
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence


# Guess rate in different conditions
guess_rates = [52. / 68., 44. / 68.] # [body-condition, world-condition]

# Implement helpers for value visualisation
map_from_action_to_subplot = lambda a: (2, 6, 8, 4)[a]
map_from_action_to_name = lambda a: ("forward", "right", "backward", "left")[a]


def smooth(x, window=10):
    return x[:window * (len(x) // window)].reshape(len(x) // window, window).mean(axis=1)


def transfer(y_temp, window=10):
    y = np.zeros((len(y_temp), len(y_temp[0])))
    for i in range(y.shape[0]):
        y[i, :] = y_temp[i]
    yline = np.zeros((y.shape[0] // window, y.shape[1]))
    for j in range(len(y_temp[0])):
        yline[:, j] = smooth(y[:, j], window=window)
    return yline


def plot_values(values, colormap='pink', vmin=-1, vmax=10):
    plt.imshow(values, interpolation="nearest", cmap=colormap, vmin=vmin, vmax=vmax)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            plt.text(x, y, '%.2f' % values[y, x], horizontalalignment='center', verticalalignment='center', color='w', fontsize=10)
    plt.yticks([])
    plt.xticks([])
    plt.colorbar(ticks=[vmin, vmax])


def plot_state_value(action_values, epsilon=0.1):
    q = action_values
    fig = plt.figure(figsize=(4, 4))
    vmin = np.min(action_values)
    vmax = max(0, np.max(action_values))
    v = (1 - epsilon) * np.max(q, axis=-1) + epsilon * np.mean(q, axis=-1)
    plot_values(v, colormap='summer', vmin=vmin, vmax=vmax)
    plt.title("$v(s)$")


def plot_action_values(action_values, epsilon=0.1):
    q = action_values
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    vmin = np.nanmin(action_values)
    vmax = max(0, np.nanmax(action_values))
    for a in [0, 1, 2, 3]:
        plt.subplot(3, 3, map_from_action_to_subplot(a))
        plot_values(q[..., a], vmin=vmin, vmax=vmax)
        action_name = map_from_action_to_name(a)
        plt.title(r"$q(s, \mathrm{" + action_name + r"})$")

    plt.subplot(3, 3, 5)
    v = (1 - epsilon) * np.nanmax(q, axis=-1) + epsilon * np.nanmean(q, axis=-1)
    plot_values(v, colormap='summer', vmin=vmin, vmax=vmax)
    plt.title("$v(s)$")


def plot_stats(stats, condition, window=10):
    xline = range(0, len(stats['episode_length']), window)

    # No arbitration
    if 'episode_td_mfQ' not in stats.keys():
        plt.figure(figsize=(8, 8))

        plt.subplot(221)
        y = stats['episode_PAP'].values
        yline = smooth(y, window=window)
        plt.plot(xline, yline, 'k')
        plt.xlim([min(xline), max(xline)])
        plt.hlines(guess_rates[condition], min(xline), max(xline), color='r', linestyle='dashed')
        plt.ylabel('Episode PAP')
        plt.xlabel('Episode Count')

        plt.subplot(222)
        yline = 0 - transfer(stats['episode_return'].values)
        plt.plot(xline, yline)
        plt.xlim([min(xline), max(xline)])
        plt.legend(['Left', 'Right'], loc='center right')
        plt.ylabel('Episode Shock')
        plt.xlabel('Episode Count')
        plt.plot(xline, np.sum(yline, axis=1), 'k', linewidth=2)

        plt.subplot(223)
        if 'episode_td_Q' in stats.keys(): # model-free algorithm
            y = stats['episode_td_Q'].values
        elif 'episode_td_SR' in stats.keys(): # model-based algorithm
            y = stats['episode_td_SR'].values
        else:
            raise ValueError('Key not available in the dictionary of episode_td')
        yline = transfer(y)
        plt.xlim([min(xline), max(xline)])
        plt.xlabel('Episode Count')
        plt.hlines(0., min(xline), max(xline), color='r', linestyle='dashed')
        if 'episode_td_Q' in stats.keys():  # model-free algorithm
            plt.plot(xline, yline)
            plt.legend(['Left', 'Right'], loc='center right')
            plt.ylabel('Episode Q-value TD')
        elif 'episode_td_SR' in stats.keys():  # model-based algorithm
            plt.plot(xline, yline, 'grey')
            plt.ylabel('Episode SR-value TD')
        plt.plot(xline, np.mean(yline, axis=1), 'k', linewidth=2)

        if 'episode_td_R' in stats.keys(): # model-based algorithm
            plt.subplot(224)
            yline = transfer(stats['episode_td_R'].values)
            plt.plot(xline, yline)
            plt.xlim([min(xline), max(xline)])
            plt.hlines(0., min(xline), max(xline), color='r', linestyle='dashed')
            plt.legend(['Left', 'Right'], loc='center right')
            plt.ylabel('Episode R-value TD')
            plt.xlabel('Episode Count')

    # With arbitration
    elif 'episode_td_mfQ' in stats.keys():
        plt.figure(figsize=(12, 8))

        plt.subplot(231)
        y = stats['episode_PAP'].values
        yline = smooth(y, window=window)
        plt.plot(xline, yline, 'k')
        plt.xlim([min(xline), max(xline)])
        plt.hlines(guess_rates[condition], min(xline), max(xline), color='r', linestyle='dashed')
        plt.ylabel('Episode PAP')
        plt.xlabel('Episode Count')

        plt.subplot(232)
        yline = 0 - transfer(stats['episode_return'].values)
        plt.plot(xline, yline)
        plt.xlim([min(xline), max(xline)])
        plt.legend(['Left', 'Right'], loc='center right')
        plt.ylabel('Episode Shock')
        plt.xlabel('Episode Count')
        plt.plot(xline, np.sum(yline, axis=1), 'k', linewidth=2)

        plt.subplot(233)
        yline = smooth(stats['episode_arb_rate_MF'].values, window=window)
        plt.plot(xline, yline)
        yline = smooth(stats['episode_arb_rate_MB'].values, window=window)
        plt.plot(xline, yline)
        plt.xlim([min(xline), max(xline)])
        plt.legend(['MF', 'MB'], loc='center right')
        plt.ylabel('Episode Arb Rate')
        plt.xlabel('Episode Count')

        plt.subplot(234)
        y = stats['episode_td_mfQ'].values
        yline = transfer(y)
        plt.xlim([min(xline), max(xline)])
        plt.xlabel('Episode Count')
        plt.hlines(0., min(xline), max(xline), color='r', linestyle='dashed')
        plt.plot(xline, yline)
        plt.legend(['Left', 'Right'], loc='center right')
        plt.ylabel('Episode MF Q-value TD')
        plt.plot(xline, np.mean(yline, axis=1), 'k', linewidth=2)

        plt.subplot(235)
        y = stats['episode_td_SR'].values
        yline = transfer(y)
        plt.xlim([min(xline), max(xline)])
        plt.xlabel('Episode Count')
        plt.hlines(0., min(xline), max(xline), color='r', linestyle='dashed')
        plt.plot(xline, yline, 'grey')
        plt.ylabel('Episode SR-value TD')
        plt.plot(xline, np.mean(yline, axis=1), 'k', linewidth=2)

        plt.subplot(236)
        yline = transfer(stats['episode_td_R'].values)
        plt.plot(xline, yline)
        plt.xlim([min(xline), max(xline)])
        plt.hlines(0., min(xline), max(xline), color='r', linestyle='dashed')
        plt.legend(['Left', 'Right'], loc='center right')
        plt.ylabel('Episode R-value TD')
        plt.xlabel('Episode Count')


def display_video(frames: Sequence[np.ndarray],
                  filename: str = 'temp.mp4',
                  frame_rate: int = 12):
    """Save and display video."""
    # Write the frames to a video.
    with imageio.get_writer(filename, fps=frame_rate) as video:
        for frame in frames:
            video.append_data(frame)

    # Read video and display the video.
    video = open(filename, 'rb').read()
    b64_video = base64.b64encode(video)
    video_tag = ('<video  width="320" height="240" controls alt="test" '
                 'src="data:video/mp4;base64,{0}">').format(b64_video.decode())
    # return IPython.display.HTML(video_tag)

