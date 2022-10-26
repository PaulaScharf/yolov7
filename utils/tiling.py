import glob
import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from shapely.geometry import Polygon

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

def tile_images_labels(path, tiles):
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
    new_path.append(path.split('/')[-1] + "_tiled")
    new_path = '/'.join(new_path)

    if not os.path.exists(new_path) or not os.path.exists(new_path + "/images") or not os.path.exists(new_path + "/labels"):
        os.makedirs(new_path)
        os.makedirs(new_path + "/images")
        os.makedirs(new_path + "/labels")
    elif len(os.listdir(new_path)) > 0:
        raise Exception("Tiling folder should be empty")

    for imname in img_files:
        imr = cv2.imread(imname, cv2.IMREAD_UNCHANGED)
        height = imr.shape[0]
        width = imr.shape[1]
        labname = '/'.join(path.split('/')[:-1]) + '/labels/' + imname.split('/')[-1].split('.')[0] + '.txt'
        labels = pd.read_csv(labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])
        
        # we need to rescale coordinates from 0-1 to real image height and width
        labels[['x1', 'w']] = labels[['x1', 'w']] * width
        labels[['y1', 'h']] = labels[['y1', 'h']] * height
        
        boxes = []
        
        # convert bounding boxes to shapely polygons. We need to invert Y and find polygon vertices from center points
        for row in labels.iterrows():
            x1 = row[1]['x1'] - row[1]['w']/2
            y1 = (height - row[1]['y1']) - row[1]['h']/2
            x2 = row[1]['x1'] + row[1]['w']/2
            y2 = (height - row[1]['y1']) + row[1]['h']/2

            boxes.append((int(row[1]['class']), Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))
        
        counter = 0
        # print('Image:', imname)
        # create tiles and find intersection with bounding boxes for each tile
        for i in range(tiles):
            for j in range(tiles):
                x1 = j*(width/tiles)
                y1 = height - (i*(height/tiles))
                x2 = ((j+1)*(width/tiles)) - 1
                y2 = (height - (i+1)*(height/tiles)) + 1

                pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                slice_labels = []


                sliced = imr[i*(height//tiles):(i+1)*(height//tiles), j*(width//tiles):(j+1)*(width//tiles)]
                sliced_im = Image.fromarray(sliced)
                filename = imname.split('/')[-1]
                slice_path = new_path + "/images/" + filename.split('/')[-1].split('.')[0] + f'_{i}_{j}.' + filename.split('.')[-1]                           
                slice_labels_path = new_path + "/labels/" + filename.split('/')[-1].split('.')[0] + f'_{i}_{j}.txt'                          
                # print(slice_path)
                sliced_im.save(slice_path)

                for box in boxes:
                    if pol.intersects(box[1]):
                        inter = pol.intersection(box[1])                  
                        
                        # get smallest rectangular polygon (with sides parallel to the coordinate axes) that contains the intersection
                        new_box = inter.envelope 
                        
                        # get central point for the new bounding box 
                        centre = new_box.centroid
                        
                        # get coordinates of polygon vertices
                        x, y = new_box.exterior.coords.xy
                        
                        # get bounding box width and height normalized to slice size
                        new_width = (max(x) - min(x)) / (width/tiles)
                        new_height = (max(y) - min(y)) / (height/tiles)
                        
                        # we have to normalize central x and invert y for yolo format
                        new_x = (centre.coords.xy[0][0] - x1) / (width/tiles)
                        new_y = (y1 - centre.coords.xy[1][0]) / (height/tiles)
                        
                        counter += 1

                        slice_labels.append([box[0], new_x, new_y, new_width, new_height])
            
                slice_df = pd.DataFrame(slice_labels, columns=['class', 'x1', 'y1', 'w', 'h'])
                # print(slice_df)
                slice_df.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')
                
    return new_path + "/images"

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]
