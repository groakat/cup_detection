import pandas as pd
import numpy as np
import pylab as plt

import pyTools.system.videoExplorer as VE

import os

import skimage.io as skio


def load_eye_tracker_data(filename):
    return pd.read_csv(filename, sep="\t", comment="#")


def load_cup_location_data(filename):
    return pd.read_csv(filename)


def filter_fixations(df):
    return df[df['B Event Info'] == "Fixation"]


def list_of_frame_numbers(df):
    return df['Frame'].tolist()


def eye_locations(df):
    return np.array([df['B POR X [px]'].tolist(), df['B POR Y [px]'].tolist()]).T


def distance(tracker_pos, target_pos):
    return np.sqrt(np.sum((tracker_pos - target_pos)**2))


def get_frame_rate(df):
    frame_numbers = list_of_frame_numbers(df)
    frames = [int(x.split(":")[-1]) for x in frame_numbers]
    return max(frames) + 1


def frames_per_hour(frame_rate):
    return frame_rate * 60 * 60


def frames_per_minute(frame_rate):
    return frame_rate * 60


def convert_i_to_frame_string(i, frame_rate):
    h = int(np.floor(i / frames_per_hour(frame_rate)))
    i -= h * frames_per_hour(frame_rate)

    m = int(np.floor(i / frames_per_minute(frame_rate)))
    i -= m * frames_per_minute(frame_rate)

    s = int(np.floor(i / frame_rate))
    f = int(i - (s * frame_rate))

    return "{:02d}:{:02d}:{:02d}:{:02d}".format(h, m, s, f)


def events_at_frame_i(df, i, frame_rate=None):
    if frame_rate is None:
        frame_rate = get_frame_rate(df)

    frame_string = convert_i_to_frame_string(i, frame_rate)
    return df[df["Frame"] == frame_string]


def fixation_at_frame_i(df, i, frame_rate=None, first_entry_only=False):
    events = events_at_frame_i(df, i, frame_rate)
    fixations = filter_fixations(events)

    if fixations.empty:
        return None
    else:
        if not first_entry_only:
            return eye_locations(fixations)
        else:
            return eye_locations(fixations.iloc[0])


def get_cup_location_at_frame_i(df, i):
    try:
        out = np.array(df[df["Frame"] == i][["cup x", "cup y"]])[0]
    except IndexError:
        out = None

    return out



def plot_tracker_target(img, tracker_pos, target_pos=None, tracker_current=True):
    if tracker_current:
        plt.scatter(tracker_pos[0], tracker_pos[1], s=100, c='b')
    else:
        plt.plot(tracker_pos[0], tracker_pos[1], 'bo', markersize=10, alpha=0.4, mec='b', mew='3')
        plt.plot(tracker_pos[0], tracker_pos[1], 'bo', markersize=10, fillstyle='none', mec='b', mew='3')
        plt.text(10, 70, "no fixation",
                 fontsize=12, bbox=dict(facecolor='w', alpha=0.5))

    if target_pos is not None:
        plt.scatter(target_pos[0], target_pos[1], s=100, c='r')


    if target_pos is not None:
        plt.plot([tracker_pos[0], target_pos[0]],
                 [tracker_pos[1], target_pos[1]])

        d = distance(tracker_pos, target_pos)
        plt.text(10, 30, "distance: {}px".format(round(d, 2)),
                 fontsize=12, bbox=dict(facecolor='w', alpha=0.5))

    plt.imshow(img)

#     return d


def save_current_figure(filename):
    ax = plt.gcf().gca()
    ax.set_axis_off()
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.savefig(filename, bbox_inches="tight", pad_inches = 0)



def tracking_on_video(video_filename, et_filename, cl_filename, out_folder):
    et_df = load_eye_tracker_data(et_filename)
    cl_df = load_cup_location_data(cl_filename)

    ve = VE.videoExplorer()
    ve.setVideoStream(video_filename, frameMode='RGB')

    p_el = np.array([0, 0])

    for i, frame in enumerate(ve):
        el = fixation_at_frame_i(et_df, i, first_entry_only=True)
        cl = get_cup_location_at_frame_i(cl_df, i)

        if el is None or min(el.ravel()) < 0:
            tracker_current = False
            el = p_el
        else:
            tracker_current = True
            p_el = el

        w, h = [x / 100.0 for x in frame.shape[:2]]
        fig = plt.figure(figsize=(h, w))

        plot_tracker_target(frame, el, cl, tracker_current=tracker_current)
        save_current_figure(os.path.join(out_folder, "frame_{:04d}.png".format(i)))
        plt.close('all')
