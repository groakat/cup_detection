import training as T
from ..features import cup as FC
from ..io import image_reader as IR

import pyTools.system.videoExplorer as VE

import numpy as np
import copy
import os
import scipy.signal as scis

def extract_features_from_video_annotation_pair(vap):
    frame_numbers = set(T.get_annotated_frames_w_numbers(vap, "card_board")["frame-numbers"])

    ve = VE.videoExplorer()
    ve.setVideoStream(vap["video_file"], frameMode='RGB')

    feats = []
    lbls = []

    for i, frame in enumerate(ve):
        feats += [FC.extract_histogram_features(frame)]

        if i in frame_numbers:
            lbls += [1]
        else:
            lbls += [0]

    return {"features": np.asarray(feats),
            "labels": np.asarray(lbls)}



def apply_smoothing(y):
    out = copy.copy(y)
    out = scis.medfilt(y, 51)

    return out


def get_card_board_frames_from_video(folder, clfr):
    ys = []

    for i, frame in enumerate(IR.images_in_path(folder)):
        feats = FC.extract_histogram_features(frame)
        ys += [clfr.predict(feats.reshape(1, -1))]

    ys = np.concatenate(ys)
    y_smooth = apply_smoothing(ys)

    return {"raw": ys,
            "smooth": y_smooth}


def get_features_from_annotation_list(annotated_video_list):
    out = []

    for avp in annotated_video_list:
        vap = T.get_video_annoation_pair(avp)
        out += [extract_features_from_video_annotation_pair(vap)]

    return out

def extract_and_save_features(annotated_video_list, out_folder):
    for i in range(len(annotated_video_list)):
        fld = get_features_from_annotation_list([annotated_video_list[i]])[0]
        filename = os.path.join(out_folder, str(i) + "_{}.npy")
        T.save_features_labels_dict(fld, filename)