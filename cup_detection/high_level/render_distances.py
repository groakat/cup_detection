from cup_detection.eye_tracker import render_distances as RD

if __name__ == "__main__":
    # eye tracker filename
    et_filename = "/Volumes/Seagate Backup Plus Drive/datasets/cups/events/CORIA/SWTS2 20150312 CORIA_table (17)_023_Trial001 Samples.txt"
    # cup location filename
    cl_filename = "/Volumes/Seagate Backup Plus Drive/datasets/cups/script_output/cup_locations_glob_raw/0_cup_locs.csv"
    # video filename
    video_filename = "/Volumes/Seagate Backup Plus Drive/datasets/cups/vt_encoded/CORIA/Scan Path_table (17)_table-23-recording/Scan Path_table (17)_table-23-recording_full.mp4"
    # folder for rendered frames
    out_folder = "/Volumes/Seagate Backup Plus Drive/datasets/cups/script_output/frames_out_video_0"

    RD.tracking_on_video(video_filename, et_filename, cl_filename, out_folder)
