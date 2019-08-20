import os
from os import listdir
import cv2
import math
import matplotlib.pyplot as plt
import skimage
import numpy as np
import gc
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize as vis
import time
from keras import backend as K
from numba import cuda
from keras.models import load_model
from multiprocessing import Pool

import tensorflow as tf

def color_white(image, mask, angle):

    h, w, _ = image.shape
    alpha = 0.5
    color = [0, 255, 0]
    color_mask = np.zeros((h, w, 3))
    color_mask[:, :, :] = color
    color_mask = image * (1-alpha) + alpha * color_mask

    if mask.shape[-1] > 0:
        one_mask = (np.sum(mask, -1, keepdims=True) >= 1)
        colored = np.where(one_mask, color_mask, image).astype(np.uint8)
        
    else:
        gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
        colored = gray.astype(np.uint8)

    _, _, num = mask.shape
    
    for i in range(num):
        yy, xx = np.where(mask[:,:,i])
        dw = max(xx) - min(xx)
        dh = max(yy) - min(yy)
        dr = np.sqrt(dw**2 + dh**2)
        xc = int(np.mean(xx))
        yc = int(np.mean(yy))
        xe = int(xc + dr/2 * math.cos(angle[i]))
        ye = int(yc + dr/2 * math.sin(angle[i]))
        colored = cv2.line(colored, (xc, yc), (xe, ye), color=[255,0,0], thickness=2)
        colored[yc-1:yc+1,xc-1:xc+1,:] = [255, 255, 255]
        xe_a1 = int(xe - 5*math.cos(angle[i] + math.pi/6))
        ye_a1 = int(ye - 5*math.sin(angle[i] + math.pi/6))
        xe_a2 = int(xe - 5*math.cos(angle[i] - math.pi/6))
        ye_a2 = int(ye - 5*math.sin(angle[i] - math.pi/6))

        colored = cv2.line(colored, (xe, ye), (xe_a1,ye_a1), color=[255,0,0], thickness=2)
        colored = cv2.line(colored, (xe, ye), (xe_a2,ye_a2), color=[255,0,0], thickness=2)

    return colored

def load_model_custom(config):

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./model')
    model.load_weights('./model/trained_model.h5', by_name=True)

    return model

class handConfig(Config):
    
    NAME = "hand"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  
    STEPS_PER_EPOCH = 1000
    DETECTION_MIN_CONFIDENCE = 0.9


class InferenceConfig(handConfig):
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    

def detect(model, img_dir):
    
    f = open("results.txt" , "w")

    for file_name in os.listdir(img_dir):
        img_path = img_dir + file_name
        print("Processing image: ", file_name)
        img_origin = skimage.io.imread(img_path)
        img = img_origin.copy()
        result = model.detect([img], verbose=0)[0]
        pred_masks = result["masks"]
        pred_orientations = result["orientations"]
        pred_bbox = result["rois"]
        print(pred_bbox)
        for box in pred_bbox:
            line = file_name + ',' + str(box[0]) + ',' + str(box[1]) + ',' + str(box[2]) + ',' + str(box[3]) + '\n'
            f.write(line)
        save_img = img_origin
        save_img = color_white(save_img, pred_masks, pred_orientations)
        skimage.io.imsave('./outputs/result_' + os.path.basename(img_path), save_img)
        print("output saved\n")
    
    f.close()

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Hand-Detector")
    parser.add_argument("--image_dir", metavar="/path/to/directory/", help="Path to the image directory")
    args = parser.parse_args()
    img_dir = args.image_dir
    model = load_model_custom(InferenceConfig())
    detect(model,img_dir)
        
