from __future__ import division

import skimage.io as skio
import skimage.segmentation as skis
import skimage.transform as skit
import skimage.color as skic
import skimage.future as skif
import skimage.morphology as skim
import skimage.measure as skime


import sklearn.ensemble as skle
import sklearn.metrics as sklm

from scipy import ndimage
import numpy as np

import pandas as pd

import os

import pyTools.videoProc.annotation as A
import pyTools.system.videoExplorer as VE

import pylab as plt



def resize_image(img):
    return skit.rescale(img, 0.25)


def min_max_normalise(color_band, r):
    return (color_band - r[0]) / (r[1] - r[0])


def normalise_lab(lab_img):
    # ranges of LAB color spaces
    # http://stackoverflow.com/a/19099064/2156909
    ranges = np.array([[0, 100],
                       [-86.185, 98.254],
                       [-107.863, 94.482]])

    out = np.zeros_like(lab_img)
    out[..., 0] = min_max_normalise(lab_img[..., 0], ranges[0])
    out[..., 1] = min_max_normalise(lab_img[..., 1], ranges[1])
    out[..., 2] = min_max_normalise(lab_img[..., 2], ranges[2])

    return out


def apply_color_model(img):
    lab_img = skic.rgb2lab(img)
    return normalise_lab(lab_img)


def segment_image(img, lab_img):
    segments = skis.slic(lab_img, compactness=30, n_segments=400)

    g = skif.graph.rag_mean_color(img, segments)
    merged_segments = skif.graph.cut_threshold(segments, g, 0.10)

    return segments + 1, merged_segments + 1 # add one to avoid a segment == 0


def center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    centre_x = (x2 - (x2 - x1) / 2) * 0.25
    centre_y = (y2 - (y2 - y1) / 2) * 0.25

    return int(centre_x), int(centre_y)


def select_cup_segment(segments, bbox):
    out = np.zeros_like(segments)
    centre_x, centre_y = center_of_bbox(bbox)

    pos_id = segments[centre_y, centre_x]
    if np.sum(segments == pos_id) < (np.prod(segments.shape) * (1 / 16)):
        out[segments == pos_id] = 1
    else:
        out[segments == pos_id] = 2

    return out


def select_negative_segments(cup_labels, segments):
    selem = skim.disk(50)
    expanded_cup_labels = skim.dilation(cup_labels > 0, selem=selem)

    neg_segment_mask = segments.copy()

    neg_segment_mask[expanded_cup_labels > 0] = 0

    return neg_segment_mask


def extract_histogram_features(frame, bins=10):
    # ranges of LAB color spaces
    # http://stackoverflow.com/a/19099064/2156909
    ranges = [[0, 100],
              [-86.185, 98.254],
              [-107.863, 94.482]]

    lab_img = skic.rgb2lab(frame.reshape(-1, 1, 3))
    H, edges = np.histogramdd(lab_img.reshape(-1, 3), bins=bins,
                                                      range=ranges)

    features = H.ravel() / np.sum(H.ravel())

    return features


def extract_features_of_segment(img, mask, bins=10):
    pixels = img[mask > 0].reshape(-1, 3)
    feat = extract_histogram_features(pixels)

    return feat


def extract_features_from_region(img, segments, region=1, global_feature=None):
    features = []
    selected_segments = segments.copy() * region
    selected_region_ids = set(np.unique(selected_segments)).difference([0])

    for r in selected_region_ids:
        segment_feature = extract_features_of_segment(img, selected_segments == r)
        if global_feature is None:
            features += [segment_feature]
        else:
            features += [np.hstack([segment_feature, global_feature])]

    return np.asarray(features)


def extract_features_of_segments(img, segments, pos_region, neg_region, global_feature=None):
    pos_features = extract_features_from_region(img, segments, pos_region, global_feature)
    neg_features = extract_features_from_region(img, segments, neg_region, global_feature)

    feat = np.concatenate([pos_features, neg_features])
    lbls = np.concatenate([np.ones((len(pos_features),)),
                           np.zeros(len(neg_features))])

    return {"features": feat,
            "labels": lbls}



def extract_features_for_training(img, bbox):
    img = resize_image(img)
    lab_img = apply_color_model(img)
    global_feature = extract_histogram_features(lab_img)

    segments, merged_segments = segment_image(img, lab_img)
    cup_labels = select_cup_segment(merged_segments, bbox)

    if np.max(cup_labels.ravel()) == 2:
        return None

    neg_segments = select_negative_segments(cup_labels, merged_segments)

    fld = extract_features_of_segments(img, segments,
                                       cup_labels > 0,
                                       neg_segments > 0,
                                       global_feature=global_feature)

    return fld


def extract_features(img):
    img = resize_image(img)
    lab_img = apply_color_model(img)
    global_feature = extract_histogram_features(lab_img)
    segments, merged_segments = segment_image(img, lab_img)

    return {"features": extract_features_from_region(img, segments,
                                                     global_feature=global_feature),
            "segments": segments,
            "merged-segments": merged_segments}

