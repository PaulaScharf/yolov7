import glob
import os
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import cv2

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

def stack_images(path, multi_frame):
    try:
        f = []  # image files
        for p in path if isinstance(path, list) else [path]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                # f = list(p.rglob('**/*.*'))  # pathlib
            elif p.is_file():  # file
                with open(p, 'r') as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                    # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
            else:
                raise Exception(f'{p} does not exist')
        img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
        assert img_files, f'No images found'
    except Exception as e:
        raise Exception(f'Error loading data from {path}')

    # Check cache
    label_files = img2label_paths(img_files)  # labels

    new_path = path.split('/')[:-1]
    new_path.append(path.split('/')[-1] + "_stacked")
    new_path = '/'.join(new_path)

    if not os.path.exists(new_path) or not os.path.exists(new_path + "/images") or not os.path.exists(new_path + "/labels"):
        os.makedirs(new_path)
        os.makedirs(new_path + "/images")
        os.makedirs(new_path + "/labels")
    elif len(os.listdir(new_path)) > 0:
        # raise Exception("Tiling folder should be empty")
        return new_path + "/images"

    for index, _ in enumerate(img_files):
        if index <= len(img_files) - multi_frame:
            index = index + multi_frame - 1
            final_img = []
            for i in range(multi_frame):
                imr = cv2.imread(img_files[index-i], cv2.IMREAD_UNCHANGED)
                # im = Image.open(img_files[index-i])
                # imr = np.array(im, dtype=np.uint8)
                if final_img == []:
                    final_img = imr
                else:
                    final_img = np.append(final_img, imr, axis=2)
            filename = img_files[index].split('/')[-1]
            stacked_path = new_path + '/images/' + filename.split('.')[0]
            np.save(stacked_path, final_img)
            shutil.copy(label_files[index], new_path + '/labels/') # TODO: what about file paths with dots
        
    return new_path + "/images"

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]
