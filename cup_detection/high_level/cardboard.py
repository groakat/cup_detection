from cup_detection.ml import cardboard as MLCB
from cup_detection.ml import training as T


if __name__ == "__main__":
    annotated_video_list = \
                    ["/Volumes/Seagate Backup Plus Drive/datasets/cups/vt_encoded/CORIA/Scan Path_table (17)_table-23-recording",
                     "/Volumes/Seagate Backup Plus Drive/datasets/cups/vt_encoded/GAIAN/Scan Path_table (1)_table-18-recording",
                     "/Volumes/Seagate Backup Plus Drive/datasets/cups/vt_encoded/GIBLE/Scan Path_table (1)_table-2-recording",
                     "/Volumes/Seagate Backup Plus Drive/datasets/cups/vt_encoded/LEVVI/Scan Path_table (1)_table-18-recording",
                     "/Volumes/Seagate Backup Plus Drive/datasets/cups/vt_encoded/PUGCE/Scan Path_room (16)_room-17-recording",
                     "/Volumes/Seagate Backup Plus Drive/datasets/cups/vt_encoded/GAIAN/Scan Path_table (2)_table-19-recording"]


    feat_folder = '/Volumes/Seagate Backup Plus Drive/datasets/cups/script_output/sub_sampled_2/'

    MLCB.extract_and_save_features(annotated_video_list, feat_folder)

    ys = T.cross_validation(feat_folder)

    print T.confusion_matrix_from_cross_validation_results(ys)

    rfc = T.train_random_forest_from_folder(feat_folder)
    out_classifier_filename = "/Volumes/Seagate Backup Plus Drive/datasets/cups/script_output/saved_classifiers/cardboard/cardboard.pkl"
    T.save_classifier(rfc, out_classifier_filename)