from utils.tiling import tile_single_image
import cv2
import numpy as np
import os
from tqdm import tqdm
import torch
import time
import argparse

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device

def applyBackgroundSubtraction(input, fgbg):
        fgmask = fgbg.apply(input)

        input = cv2.cvtColor(input,cv2.COLOR_BGR2BGRA)
        input[:, :, 3] = fgmask

        return input, fgbg

def normalizeIllumination(input, init_avs, a):
    current_av = np.mean(np.mean(input,axis=1),axis=0)
    init_avs.append(current_av)
    if len(init_avs) > 20:
        init_avs.pop(0)
    init_av = np.mean(init_avs, axis=0)

    input = (np.clip(init_av + a*(input - current_av), 0, 255)).astype('uint8')
    return input, init_avs


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def measure_inference(source, weights, mode, device_type):

    init_phase = 3

    img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo', 'npy']  # acceptable image suffixes

    if mode != 1:
        init_phase = 0

    layers = (4 if mode==1 else 3)
    multi_frame = (3 if mode==2 else 1)

    first=0

    if mode == 1:
        # background subtraction
        fgbg = cv2.createBackgroundSubtractorMOG2(history = 50, varThreshold = 50, detectShadows = False)
        # illumination normalisation
        init_avs = []
        a=1

    device = select_device(device_type)
    print(device.type)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(4608, s=stride)  # check img_size
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    if half:
        model.half()  # to FP16

    if device.type != 'cpu':
        model(torch.zeros(1, multi_frame*layers, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    files = os.listdir(source)
    files.sort()

    first=0

    inference_times = []
    backsub_times = []

    for filename in files[:500]:
        source_path = os.path.join(source, filename)
        if os.path.isfile(source_path) and (filename.endswith(".JPG") or filename.endswith(".png") or filename.endswith(".npy")):
            if filename.endswith(".npy"):
                img = np.load(source_path)
            else:
                img = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)

            t1 = time_synchronized()*1000.0

            if mode == 1:
                img, init_avs = normalizeIllumination(img, init_avs, a)
                img, fgbg = applyBackgroundSubtraction(img, fgbg)
                t2 = time_synchronized()*1000.0

                print(f'({(t2 - t1):.1f}ms) Background Subtraction')
                backsub_times.append(t2-t1)

            if first<init_phase:
                first+=1
            else:
                img = torch.from_numpy(img.transpose(2, 0, 1)).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Warmup
                if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                    old_img_b = img.shape[0]
                    old_img_h = img.shape[2]
                    old_img_w = img.shape[3]
                    for i in range(3):
                        model(img, augment=False)[0]

                # Inference
                pred = model(img, augment=False)[0]
                pred = non_max_suppression(pred, 0.26, 0.5)
                t3 = time_synchronized()*1000.0

                print(f'({(t3 - t1):.1f}ms) Inference')
                inference_times.append(t3-t1)

    print(f'({((np.mean(inference_times))):.1f}ms) Average Inference') 
    print(f'({((np.std(inference_times))):.1f}ms) STD Inference') 
    print(f'({((np.var(inference_times))):.1f}ms) Variance Inference') 
    print('\n')
    print(f'({((np.mean(backsub_times))):.1f}ms) Average Background Subtraction') 
    print(f'({((np.std(backsub_times))):.1f}ms) STD Background Subtraction') 
    print(f'({((np.var(backsub_times))):.1f}ms) Variance Background Subtraction') 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/home/paula/documents/masterthesis/test-pipeline/results/weights/alpha.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='/home/paula/documents/masterthesis/test-pipeline/tempi/alpha', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--mode', type=int, default=1, choices=range(0,3), help='0=plain, 1=background, 2=stacking')
    parser.add_argument('--device', type=str, default='0', help='device')
    
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    device = select_device('')
    print(device.type)

    # with torch.no_grad():
    #     measure_inference(opt.source, opt.weights, opt.mode, opt.device)
