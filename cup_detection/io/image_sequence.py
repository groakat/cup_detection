import os
import shutil
import subprocess


def clear_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.makedirs(folder)


def video_to_image_sequence(video_file, output_folder):
    clear_folder(output_folder)
    out = subprocess.Popen("ffmpeg -i '{}' '{}'".format(video_file,
                                            os.path.join(output_folder, "%06d.png")), shell=True).communicate()[0]
    print(out)


def image_sequence_to_video(input_folder, video_file):
    out = subprocess.Popen("ffmpeg -i '{}' '{}'".format(os.path.join(input_folder, "%06d.png"),
                                                        video_file), shell=True).communicate()[0]

    print (out)
