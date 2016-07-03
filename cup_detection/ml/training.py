from __future__ import division

import sklearn.ensemble as skle
import sklearn.metrics as sklm
from sklearn.externals import joblib
import numpy as np

import os

import pyTools.videoProc.annotation as A


def get_video_annoation_pair(folder):
    files = os.listdir(folder)
    vap = {}

    for f in files:
        if f.endswith(".csv"):
            vap["anno_file"] = os.path.join(folder, f)
        elif f.endswith("_full.mp4"):
            vap["video_file"] = os.path.join(folder, f)

    return vap


def get_annotated_frames_w_numbers(vap):
    anno = A.Annotation()
    anno.loadFromFile(vap["anno_file"])

    filtered_anno = anno.getFramesWithBehaviour("cup")

    return {"frame-numbers": A.getFramesFromFrameAnno(filtered_anno),
            "filtered-anno": filtered_anno}



def contains_nan(x):
    """ http://stackoverflow.com/a/6736970/2156909 """
    return np.isnan(np.sum(x))


def save_features_labels_dict(fld, filename):
    """
    filename: (string) with placeholder for `format` to insert "feats" and "lbls"
    """
    np.save(filename.format("feat"), fld['features'])
    np.save(filename.format("lbls"), fld['labels'])


def load_features_labels_dict(filename):
    """
    filename: (string) with placeholder for `format` to insert "feats" and "lbls"
    """
    out = {}
    out['features'] = np.load(filename.format("feat"))
    out['labels'] = np.load(filename.format("lbls"))

    return out



def subsample_features(fld, step_size):
    """ subssample frames of features to save space and training time
    fld: feature-labels-dict
    """

    out = {}
    out['features'] = fld['features'][::step_size]
    out['labels'] = fld['labels'][::step_size]

    return out


def features_labels_dicts_in_folder(folder):
    files = os.listdir(folder)

    out = []
    for f in files:
        if f.endswith("feat.npy"):
            filename = os.path.join(folder, f[:-8])
            filename += "{}.npy"

            if os.path.basename(filename).format("lbls") in files:
                out += [filename]

    return sorted(out)


def train_test_split(folder, training_ids):
    files = features_labels_dicts_in_folder(folder)
    files = np.asarray(files)

    testing_ids = set(range(len(files)))
    testing_ids = list(testing_ids.difference(training_ids))

    return {"train": files[training_ids],
            "test": files[testing_ids]}


def load_and_concatenate_features_labels_dict(fld_file_list):
    feats = []
    lbls = []

    for fld_file in fld_file_list:
        fld = load_features_labels_dict(fld_file)
        feats += [fld['features']]
        lbls += [fld['labels']]

    return {"features": np.concatenate(feats),
            "labels": np.concatenate(lbls)}


def train_random_forest_from_file_list(fld_file_list):
    fld = load_and_concatenate_features_labels_dict(fld_file_list)

    rfc = skle.RandomForestClassifier(n_estimators=20, n_jobs=4)
    rfc.fit(fld['features'], fld['labels'])

    return rfc


def train_random_forest_w_split(folder, training_ids):
    fld_filelist = train_test_split(folder, training_ids)
    rfc = train_random_forest_from_file_list(fld_filelist['train'])

    return rfc


def train_random_forest_from_folder(folder):
    training_ids = list(range(len(features_labels_dicts_in_folder(folder))))
    rfc = train_random_forest_w_split(folder, training_ids)

    return rfc


def save_classifier(clf, filename):
    joblib.dump(clf, filename)


def load_classifier(filename):
    return joblib.load(filename)


def test_random_forest_w_split(folder, training_ids, rfc):
    fld_filelist = train_test_split(folder, training_ids)
    fld = load_and_concatenate_features_labels_dict(fld_filelist['test'])

    y = rfc.predict(fld['features'])

    return {"predict": y,
            "ground-truth": fld['labels']}


def cross_validation(folder):
    num_videos = set(range(len(features_labels_dicts_in_folder(folder))))

    res = []

    for i in sorted(num_videos):
        training_ids = list(num_videos.difference([i]))
        rfc = train_random_forest_w_split(folder, training_ids)
        res += [test_random_forest_w_split(folder, training_ids, rfc)]

    return res


def confusion_matrix_from_cross_validation_results(ys):
    pred = []
    gt = []

    for i, y in enumerate(ys):
        print "partial confusion matrix for video {}".format(i)
        print sklm.confusion_matrix(y['predict'], y['ground-truth'])
        print "------------------------------------"

        pred += [y['predict']]
        gt += [y['ground-truth']]

    pred = np.concatenate(pred)
    gt = np.concatenate(gt)

    return sklm.confusion_matrix(pred, gt)