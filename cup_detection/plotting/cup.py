from __future__ import division

import skimage.color as skic
import numpy as np

import pyTools.system.videoExplorer as VE

import pylab as plt

from ..ml import cup as MLC
from ..features import cup as FC
from ..ml import training as T


def plot_cup(img, bbox):
    img = FC.resize_image(img)
    lab_img = FC.apply_color_model(img)

    labels1, labels2 = FC.segment_image(img, lab_img)

    cup_labels = FC.select_cup_segment(labels2, bbox)
#     out3 = skic.label2rgb(cup_labels, img, kind='avg')
    out3 = cup_labels

    neg_segments = FC.select_negative_segments(cup_labels, labels2)
    out4 = skic.label2rgb(neg_segments, img, kind='avg')

    plt.figure()
    plt.imshow(img)
    plt.title("Image")
    plt.figure()
    plt.imshow(skic.label2rgb(labels1, img, kind='avg'))
    plt.title("Plain SLIC result")
    plt.figure()
    plt.imshow(skic.label2rgb(labels2, img, kind='avg'))
    plt.title("Merged segments")
    plt.figure()
    plt.imshow(out3)
    if np.max(out3.ravel()) == 1:
        plt.title("cup only")
    else:
        plt.title("rejected cup")

    plt.figure()
    plt.imshow(out4)
    plt.title("Negative segments")

#     plt.plot(x, y)
#     print x, y


def plot_cup_prediction(frame, segments, lbls):
    centroids = MLC.get_centroids(segments)

    pos_loc = np.where(lbls)[0]

    plt.figure(figsize=(20,10))
    plt.imshow(frame)
    plt.scatter(centroids[:, 1], centroids[:, 0], c=lbls[:, 1], cmap=plt.cm.viridis)


def plot_random_cup_predictions(vap, classifier):
    d = T.get_annotated_frames_w_numbers(vap)
    frame_numbers = list(d['frame-numbers'])
    bboxs = np.asarray(d['filtered-anno'])[:, :4]

    selected_frames = np.random.permutation(np.arange(len(frame_numbers)))[:10]

    ve = VE.videoExplorer()
    ve.setVideoStream(vap["video_file"], frameMode='RGB')

    feats = []
    lbls = []

    for i in selected_frames:
        frame = ve.getFrame(vap["video_file"], frame_numbers[i], frameMode='RGB')
        fsd = FC.extract_features(frame) # fsd: feature_segment_dict

        feats = fsd['features']
        lbls = classifier.predict_proba(feats)

        print np.sum(lbls)

        segments = fsd['segments']
        plot_cup_prediction(frame, segments, lbls)
        plt.title("frame {} of {}".format(i, vap["video_file"]))


def plot_random_predictions_from_folder(folder, annotated_video_list):
    num_videos = set(range(len(T.features_labels_dicts_in_folder(folder))))

    for i in sorted(num_videos):
        training_ids = list(num_videos.difference([i]))
        rfc = T.train_random_forest_w_split(folder, training_ids)
        vap = T.get_video_annoation_pair(annotated_video_list[i])
        plot_random_cup_predictions(vap, rfc)



def plot_cup_selection(vap):
    d = T.get_annotated_frames_w_numbers(vap)
    frame_numbers = list(d['frame-numbers'])
    bboxs = np.asarray(d['filtered-anno'])[:, :4]

    selected_frames = np.random.permutation(np.arange(len(frame_numbers)))[:10]

    ve = VE.videoExplorer()
    ve.setVideoStream(vap["video_file"], frameMode='RGB')

    feats = []
    lbls = []

    for i in selected_frames:
        frame = ve.getFrame(vap["video_file"], frame_numbers[i], frameMode='RGB')
        plot_cup(frame, bboxs[i])