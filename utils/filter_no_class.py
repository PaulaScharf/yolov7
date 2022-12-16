import os
import random
import shutil
from tqdm import tqdm

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo', '.npy']  # acceptable image suffixes

def filter_no_class(path, max_perc):
    print('[INFO] maximum percentage of images without detections: ' + str(max_perc))
    new_path = path.split('/')[:-1]
    new_path.append(path.split('/')[-1] + "_filtered")
    new_path = '/'.join(new_path)
    if not os.path.exists(new_path) or not os.path.exists(new_path + "/images") or not os.path.exists(new_path + "/labels"):
        os.makedirs(new_path)
        os.makedirs(new_path + "/images")
        os.makedirs(new_path + "/labels")
    elif len(os.listdir(new_path)) > 0:
        # raise Exception("Filtered folder should be empty")
        print('[INFO] reuse existing filtered images')
        return new_path + "/images"
    
    labels_path = '/'.join(path.split('/')[:-1]) + '/labels/'
    labels_files = os.listdir(labels_path)
    total_labels = len([name for name in labels_files if os.path.isfile(labels_path + name)])
    empty_labels = len([name for name in labels_files if os.path.isfile(labels_path + name) and os.stat(labels_path + name).st_size == 0])
    not_empty_labels = total_labels - empty_labels

    applied_perc = not_empty_labels/((empty_labels/max_perc)-empty_labels) if max_perc > 0 else max_perc

    if applied_perc <= 1.0:
        removed = 0
        kept = 0
        print('[INFO] Filtering...')
        pbar = enumerate(os.listdir(path))
        pbar = tqdm(pbar, total=len(os.listdir(path)))  # progress bar

        for _, name in pbar:
            label_path = os.path.join(labels_path, name[:-4] + ".txt")
            img_path = os.path.join(path, name)
            if os.path.isfile(label_path):
                random_perc = random.randint(0,10000000)/10000000
                if random_perc <= applied_perc or os.stat(label_path).st_size > 0:
                    shutil.copy(label_path, new_path + '/labels/')
                    shutil.copy(img_path, new_path + '/images/')
                    kept += 1
                else:
                    removed += 1 
        
        print('[INFO] removed ' + str(removed) + ' images')
        print('[INFO] kept ' + str(kept) + ' images')

        total_labels = len([name for name in os.listdir(new_path + '/labels/') if os.path.isfile(new_path + '/labels/' + name)])
        empty_labels = len([name for name in os.listdir(new_path + '/labels/') if os.path.isfile(new_path + '/labels/' + name) and os.stat(new_path + '/labels/' + name).st_size == 0])
        print('[INFO] resulting percentage of images without classes: ' + str(empty_labels/total_labels))
        return new_path + '/images/'
    else:
        print('[INFO] percentage of images without classes: ' + str(empty_labels/total_labels))
        return path

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]