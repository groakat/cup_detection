from cup_detection.ml import cup as MLC
from cup_detection.ml import training as T
from cup_detection.ml import dataset_curration as DC

import os


def extract_and_save_features(annotated_video_list, out_folder):
    for i in range(len(annotated_video_list))[:1]:
        fld = MLC.get_features_from_annotation_list([annotated_video_list[i]])[0]
        filename = os.path.join(out_folder, str(i) + "_{}.npy")
        T.save_features_labels_dict(fld, filename)

if __name__ == "__main__":
    annotated_video_list = \
                    ["/Volumes/Seagate Backup Plus Drive/datasets/cups/vt_encoded/CORIA/Scan Path_table (17)_table-23-recording",
                     "/Volumes/Seagate Backup Plus Drive/datasets/cups/vt_encoded/GAIAN/Scan Path_table (1)_table-18-recording",
                     "/Volumes/Seagate Backup Plus Drive/datasets/cups/vt_encoded/GIBLE/Scan Path_table (1)_table-2-recording",
                     "/Volumes/Seagate Backup Plus Drive/datasets/cups/vt_encoded/LEVVI/Scan Path_table (1)_table-18-recording",
                     "/Volumes/Seagate Backup Plus Drive/datasets/cups/vt_encoded/PUGCE/Scan Path_room (16)_room-17-recording",
                     "/Volumes/Seagate Backup Plus Drive/datasets/cups/vt_encoded/GAIAN/Scan Path_table (2)_table-19-recording"]


    out_folder = "/Volumes/Seagate Backup Plus Drive/datasets/cups/script_output/cup_features_glob_raw_all_test/"

    extract_and_save_features(annotated_video_list, out_folder)

    DC.select_neg_features_from_datasets('/Volumes/Seagate Backup Plus Drive/datasets/cups/script_output/cup_features_glob_raw_all_test/',
                                         '/Volumes/Seagate Backup Plus Drive/datasets/cups/script_output/cup_features_glob_raw_sel_test/')
    #

    folder = '/Volumes/Seagate Backup Plus Drive/datasets/cups/script_output/cup_features_glob_raw_sel/'
    ys = T.cross_validation(folder)

    print T.confusion_matrix_from_cross_validation_results(ys)