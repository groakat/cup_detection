import os
import glob
import skimage.io as skio


def get_image_paths(folder):
    images = sorted(glob.glob(os.path.join(folder, "*.png")))

    return images


def images_in_path(folder):
    images = get_image_paths(folder)

    for img in images:
        yield skio.imread(img)
