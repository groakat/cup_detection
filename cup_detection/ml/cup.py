from __future__ import division

import numpy as np
import pandas as pd
import os

import skimage.measure as skime


import pyTools.system.videoExplorer as VE

import training as T
from ..features import cup as FC




def extract_features_from_video_annotation_pair(vap):
    d = T.get_annotated_frames_w_numbers(vap, "cup")
    frame_numbers = list(d['frame-numbers'])
    bboxs = np.asarray(d['filtered-anno'])[:, :4]

    ve = VE.videoExplorer()
    ve.setVideoStream(vap["video_file"], frameMode='RGB')

    feats = []
    lbls = []

    # counter for bbox index
    cnt = 0
    for i, frame in enumerate(ve):
        if i not in frame_numbers:
            continue

        if T.contains_nan(bboxs[cnt]):
            cnt += 1
            continue

        fld = FC.extract_features_for_training(frame, bboxs[cnt])
        cnt += 1

        if fld is None:
            # cup present, but could not get segmented reliably
            continue

        feats += [fld['features']]
        lbls += [fld['labels']]


    return {"features": np.concatenate(feats),
            "labels": np.concatenate(lbls)}




def get_features_from_annotation_list(annotated_video_list):
    out = []

    for avp in annotated_video_list:
        vap = T.get_video_annoation_pair(avp)
        out += [extract_features_from_video_annotation_pair(vap)]

    return out


def get_centroids(segments, resize_multiplier=4):
    """
    resize_multiplier corrects for scaling the frames down for feature extraction
    """
    rp = skime.regionprops(segments)
    centroids = np.array([x['centroid'] for x in rp]) * resize_multiplier

    return centroids


def confidence_of_merged_region(lbls, segments, merged_segments):
    scores = np.zeros((np.max(merged_segments.ravel()) + 1,))
    counts = np.zeros((np.max(merged_segments.ravel()) + 1,))

    centroids = get_centroids(segments, resize_multiplier=1).astype(np.int)

    for i, c in enumerate(centroids):
        merged_region = merged_segments[c[0], c[1]]
        scores[merged_region] += lbls[i, 1]
        counts[merged_region] += 1

#     counts[counts == 0] = 1
#     scores /= counts

    scores[counts > 40] = 0

    return scores


def predict_cup_center(frame, classifier):
    fsd = FC.extract_features(frame) # fsd: feature_segment_dict

    feats = fsd['features']
    lbls = classifier.predict_proba(feats)

    segments = fsd['segments']
    merged_segments = fsd['merged-segments']
    merged_seg_centroids = get_centroids(merged_segments)

    confidences = confidence_of_merged_region(lbls, segments, merged_segments)
    highest_confidence = np.argmax(confidences)

    return {"winner": merged_seg_centroids[highest_confidence - 1],
            "confidences": confidences,
            "centroids": merged_seg_centroids}


def cup_predictions(vap, classifier):
    d = T.get_annotated_frames_w_numbers(vap, "cup")
    frame_numbers = list(d['frame-numbers'])
    bboxs = np.asarray(d['filtered-anno'])[:, :4]

    selected_frames = frame_numbers

    ve = VE.videoExplorer()
    ve.setVideoStream(vap["video_file"], frameMode='RGB')

    data = []

    for i, frame_no in enumerate(selected_frames):
        if i == 0 or (selected_frames[i - 1] + 1) != selected_frames[i]:
            frame = ve.getFrame(vap["video_file"], frame_no, frameMode='RGB')
        else:
            frame = ve.next()

        ccp = predict_cup_center(frame, classifier)

        confidences = ccp["confidences"]
        centroids = ccp["centroids"]
        cc = ccp["winner"]

        data += [[frame_no, cc[1], cc[0]]]

    df = pd.DataFrame(data=data, columns=["Frame", "cup x", "cup y"])

    return df


def cup_predictions_from_folder(folder, annotated_video_list, out_folder, classifiers=None):
    num_videos = set(range(len(T.features_labels_dicts_in_folder(folder))))

    if classifiers is None:
        classifiers = []

    for i in sorted(num_videos):
        training_ids = list(num_videos.difference([i]))
        if len(classifiers) <= i:
            rfc = T.train_random_forest_w_split(folder, training_ids)
            classifiers += [rfc]

        vap = T.get_video_annoation_pair(annotated_video_list[i])
        df = cup_predictions(vap, classifiers[i])

        df.to_csv(os.path.join(out_folder, "{}_cup_locs.csv".format(i)))


def extract_and_save_features(annotated_video_list, out_folder):
    for i in range(len(annotated_video_list)):
        fld = get_features_from_annotation_list([annotated_video_list[i]])[0]
        filename = os.path.join(out_folder, str(i) + "_{}.npy")
        T.save_features_labels_dict(fld, filename)