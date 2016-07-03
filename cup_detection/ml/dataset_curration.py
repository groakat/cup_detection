from __future__ import division

import numpy as np
import os

import training as T

def select_neg_features_from_dataset(in_file, max_multipier=10):
    fld = T.load_features_labels_dict(in_file)
    feats = fld['features']
    lbls = fld['labels']

    neg_lbls = np.where(lbls == 0)[0][::2]
    pos_lbls = np.where(lbls == 1)[0][::2]
    selected_lbls = np.random.permutation(neg_lbls)[:len(pos_lbls) * max_multipier]
    selected_lbls = np.concatenate([selected_lbls, pos_lbls])

    return {"features": feats[selected_lbls],
            "labels": lbls[selected_lbls]}

def select_neg_features_from_datasets(in_folder, out_folder):
    files = T.features_labels_dicts_in_folder(in_folder)

    for f in files:
        fld = select_neg_features_from_dataset(f)
        out_file = os.path.join(out_folder, os.path.basename(f))
        T.save_features_labels_dict(fld, out_file)

