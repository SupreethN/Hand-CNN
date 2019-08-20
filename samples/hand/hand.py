import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt
import pickle
# Root directory of the project
# ROOT_DIR = os.path.abspath("../../")
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize as vis
import cv2
import math
import scipy
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "model/mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "new_logs")

############################################################
#  Configurations
############################################################

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


class handConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "hand"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + hand

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.90


############################################################
#  Dataset
############################################################

class HandDataset(utils.Dataset):

    def load_hand(self, dataset_file):
        """Load a subset of the hand dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("hand", 1, "hand")

        # Train or validation dataset?

        # We mostly care about the x and y coordinates of each region
        file = open(dataset_file)
        lines = file.readlines()
        anno_dict = {}
        for line in lines:
            line_split = line.split(",")
            if line_split[0] not in anno_dict:
                anno_dict[line_split[0]] = []
            anno_dict[line_split[0]].append([int(x) for x in line_split[5:-1]])

        # Add images
        for idx, image_path in enumerate(anno_dict):
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance.
            polygons = anno_dict[image_path]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "hand",
                image_id=idx,
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a hand dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "hand":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, points in enumerate(info["polygons"]):
            # print(i, points)
            x = [points[0], points[2], points[4], points[6]]
            y = [points[1], points[3], points[5], points[7]]
            for inx in range(4):
                if x[inx] >= info["width"]:
                    x[inx] = info["width"] - 1
                if y[inx] >= info["height"]:
                    y[inx] = info["height"] - 1
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(y, x)
            mask[rr, cc, i] = 1
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def load_orientations(self, image_id):
        """Generate instance orientaion for an image.
       Returns:
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "hand":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        orientation = np.zeros(shape=(len(info["polygons"]), 1), dtype=np.float32)

        for i, points in enumerate(info["polygons"]):
            x = [points[0], points[2], points[4], points[6]]
            y = [points[1], points[3], points[5], points[7]]
            x_wrist = (x[0] + x[1]) / 2
            y_wrist = (y[0] + y[1]) / 2
            x_hand = (x[2] + x[3]) / 2
            y_hand = (y[2] + y[3]) / 2
            x_delta = x_hand - x_wrist
            y_delta = y_hand - y_wrist
            orientation[i, 0] = math.atan2(y_delta, x_delta)
        return orientation

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "hand":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = HandDataset()
    dataset_train.load_hand(args.dataset + "train.txt")
    dataset_train.prepare()
    #
    # Validation dataset
    dataset_val = HandDataset()
    dataset_val.load_hand(args.dataset + "oxford_annos.txt")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='all')



def test(model, to_save=True):
    """

    :param model:
    :param to_save: save the result image
    :return:
    """
    # Test dataset.
    testset = args.testset
    dataset_test = HandDataset()

    testset == "oxford"
    dataset_test.load_hand(args.dataset + "oxford_annos.txt")


    dataset_test.prepare()
    pred_m = []
    gt_m = []
    pred_s = []
    dir_name = "./samples/hand/results/"+testset+"_{:%m%d%H%M}/".format(datetime.datetime.now())
    if to_save:
        if not os._exists(dir_name):
            os.mkdir(dir_name)
    gt_a = []
    pred_a = []
    for image_info in dataset_test.image_info:
        print(image_info)
        image_id = image_info['id']
        image_path = image_info['path']
        img_origin = skimage.io.imread(image_path)
        h, w, _ = img_origin.shape
        img = img_origin.copy()	

        gt_polygons = image_info['polygons']
        gt_boxes = []
        gt_class_ids = []

        for gt_polygon in gt_polygons:
            x = [gt_polygon[0], gt_polygon[2], gt_polygon[4], gt_polygon[6]]
            y = [gt_polygon[1], gt_polygon[3], gt_polygon[5], gt_polygon[7]]
            gt_boxes.append([min(y), min(x), max(y), max(x)])
            gt_class_ids.append(1)
        gt_boxes = np.array(gt_boxes)
        gt_class_ids = np.array(gt_class_ids)
        gt_masks, gt_mask_class_ids = dataset_test.load_mask(image_id)
        gt_orientations = dataset_test.load_orientations(image_id)

        result = model.detect([img], verbose=0)[0]
        pred_boxes = result['rois']
        pred_class_ids = result["class_ids"]
        pred_scores = result["scores"]
        pred_masks = result["masks"]
        pred_orientations = result["orientations"]
        save_img = img_origin
        y1 = -1
        for gt_box in gt_boxes:
            y1, x1, y2, x2 = gt_box
        if y1 > 0:
            if len(pred_boxes) > 0:
                gt_match, pred_match, overlaps, pred_scores, gt_angles, pred_angles = \
                            utils.compute_matches_with_scores(gt_boxes, gt_class_ids, gt_masks, gt_orientations,
                            pred_boxes, pred_class_ids, pred_scores, pred_masks, pred_orientations,
                            iou_threshold=0.5, score_threshold=0.0)
                gt_a.extend(gt_angles)
                pred_a.extend(pred_angles)
                if to_save:
                    save_img = color_white(save_img, pred_masks, pred_orientations)
            else:
                gt_match = len(gt_boxes) * [-1]
                pred_match = []
                pred_scores = []
        else:
            gt_match = []
            if len(pred_boxes) > 0:
                pred_match = len(pred_boxes) * [-1]
                pred_scores = pred_scores
            else:
                pred_match = []
                pred_scores = []
        if to_save:
            filename = dir_name + image_path.split('/')[-1]
            print(filename)
            skimage.io.imsave(filename, save_img)

        print("pred_match: ", pred_match)
        print("gt_match: ", gt_match)
        print("pred_scores",pred_scores)
        gt_m.extend(gt_match)
        pred_m.extend(pred_match)
        pred_s.extend(pred_scores)

        # Temp AP
        assert len(pred_m) == len(pred_s)
        tmp_pred_m = np.array(pred_m)
        tmp_gt_m = np.array(gt_m)
        tmp_pred_s = np.array(pred_s)
        # sort the score
        tmp_sorted_idx = np.argsort(tmp_pred_s)[::-1]
        tmp_pred_m = tmp_pred_m[tmp_sorted_idx]
        # Compute precision and recall at each prediction box step 
        tmp_precisions = np.cumsum(tmp_pred_m > -1) / (np.arange(len(tmp_pred_m)) + 1)
        tmp_recalls = np.cumsum(tmp_pred_m > -1).astype(np.float32) / len(tmp_gt_m)
        print("AP = ", voc_ap(tmp_recalls, tmp_precisions))

    # Compute mean AP over recall range
    assert len(pred_m) == len(pred_s)
    pred_m = np.array(pred_m)
    gt_m = np.array(gt_m)
    pred_s = np.array(pred_s)
    # sort the score
    sorted_idx = np.argsort(pred_s)[::-1]
    pred_m = pred_m[sorted_idx]
    pred_s = pred_s[sorted_idx]
    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_m > -1) / (np.arange(len(pred_m)) + 1)
    recalls = np.cumsum(pred_m > -1).astype(np.float32) / len(gt_m)
    precision_recall_hand_rcnn = np.concatenate((precisions, recalls), axis=0)
    
    mAP = voc_ap(recalls, precisions)
    print("AP = ", mAP)

    pr_dict = {"precison":precisions, "recall":recalls}

    # angle
    delta_angles = [np.abs(pred_a[i] - gt_a[i]) for i in range(len(pred_a))]
    for i in range(len(delta_angles)):
        delta_angles[i] = delta_angles[i] % 360
        if delta_angles[i] > 180:
            delta_angles[i] = 360 - delta_angles[i]

    def angle_accuracy(d_angles, thres=10):
        pred_r = [dangle <= thres for dangle in d_angles]
        accu = sum(pred_r) / len(pred_r)
        return accu,

    accuracys = [angle_accuracy(delta_angles, thres) for thres in range(90)]
    print("num matched = ", len(delta_angles))
    print("thres = 10, accu = ", accuracys[10])
    print("thres = 20, accu = ", accuracys[20])
    print("thres = 30, accu = ", accuracys[30])

    
    return mAP


def test_bbox(model, to_save=True):
    # Test dataset.
    testset = "oxford"
    dataset_test = HandDataset()
    dataset_test.load_hand(args.dataset + "oxford_annos.txt")
    dataset_test.prepare()
    pred_m = []
    gt_m = []
    pred_s = []
    dir_name = "./samples/hand/results/oxford_{:%m%d%H%M}/".format(
        datetime.datetime.now())
    if to_save:
        if not os._exists(dir_name):
            os.mkdir(dir_name)
    gt_a = []
    pred_a = []
    for image_info in dataset_test.image_info:
        print(image_info)
        image_id = image_info['id']
        image_path = image_info['path']
        img_origin = skimage.io.imread(image_path)
        h, w, _ = img_origin.shape
        img = img_origin.copy()

        gt_polygons = image_info['polygons']
        gt_boxes = []
        gt_class_ids = []

        for gt_polygon in gt_polygons:
            x = [gt_polygon[0], gt_polygon[2], gt_polygon[4], gt_polygon[6]]
            y = [gt_polygon[1], gt_polygon[3], gt_polygon[5], gt_polygon[7]]
            gt_boxes.append([min(y), min(x), max(y), max(x)])
            gt_class_ids.append(1)
        gt_boxes = np.array(gt_boxes)
        gt_class_ids = np.array(gt_class_ids)
        gt_masks, gt_mask_class_ids = dataset_test.load_mask(image_id)
        gt_orientations = dataset_test.load_orientations(image_id)

        result = model.detect([img], verbose=0)[0]
        pred_boxes = result['rois']
        pred_class_ids = result["class_ids"]
        pred_scores = result["scores"]
        pred_masks = result["masks"]
        pred_orientations = result["orientations"]
        save_img = img_origin
        y1 = -1
        for gt_box in gt_boxes:
            y1, x1, y2, x2 = gt_box
        if y1 > 0:
            if len(pred_boxes) > 0:
                gt_match, pred_match, overlaps, pred_scores, gt_angles, pred_angles = \
                    utils.compute_matches_with_scores_bbox(gt_boxes, gt_class_ids, gt_masks, gt_orientations,
                                                      pred_boxes, pred_class_ids, pred_scores, pred_masks,
                                                      pred_orientations,
                                                      iou_threshold=0.5, score_threshold=0.0)
                gt_a.extend(gt_angles)
                pred_a.extend(pred_angles)

                if to_save:
                    save_img = color_white(save_img, pred_masks, pred_orientations)
            else:
                gt_match = len(gt_boxes) * [-1]
                pred_match = []
                pred_scores = []
        else:
            gt_match = []
            if len(pred_boxes) > 0:
                pred_match = len(pred_boxes) * [-1]
                pred_scores = pred_scores
            else:
                pred_match = []
                pred_scores = []
        if to_save:
            filename = dir_name + image_path.split('/')[-1]
            print(filename)
            skimage.io.imsave(filename, save_img)

        print("pred_match: ", pred_match)
        print("gt_match: ", gt_match)
        print("pred_scores", pred_scores)
        gt_m.extend(gt_match)
        pred_m.extend(pred_match)
        pred_s.extend(pred_scores)
        # Temp AP
        assert len(pred_m) == len(pred_s)
        tmp_pred_m = np.array(pred_m)
        tmp_gt_m = np.array(gt_m)
        tmp_pred_s = np.array(pred_s)
        # sort the score
        tmp_sorted_idx = np.argsort(tmp_pred_s)[::-1]
        tmp_pred_m = tmp_pred_m[tmp_sorted_idx]
        # Compute precision and recall at each prediction box step
        tmp_precisions = np.cumsum(tmp_pred_m > -1) / (np.arange(len(tmp_pred_m)) + 1)
        tmp_recalls = np.cumsum(tmp_pred_m > -1).astype(np.float32) / len(tmp_gt_m)
        print("AP = ", voc_ap(tmp_recalls, tmp_precisions))

    # Compute mean AP over recall range
    assert len(pred_m) == len(pred_s)
    pred_m = np.array(pred_m)
    gt_m = np.array(gt_m)
    pred_s = np.array(pred_s)
    # sort the score
    sorted_idx = np.argsort(pred_s)[::-1]
    pred_m = pred_m[sorted_idx]
    pred_s = pred_s[sorted_idx]
    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_m > -1) / (np.arange(len(pred_m)) + 1)
    recalls = np.cumsum(pred_m > -1).astype(np.float32) / len(gt_m)
    mAP = voc_ap(recalls, precisions)
    print("AP = ", mAP)

    plt.figure(1)
    plt.plot(recalls,precisions)
    plt.savefig(dir_name+args.testset+"_pre_rec.png")

    pr_dict = {"precison":precisions, "recall":recalls}

    # angle
    delta_angles = [np.abs(pred_a[i] - gt_a[i]) for i in range(len(pred_a))]
    for i in range(len(delta_angles)):
        delta_angles[i] = delta_angles[i] % 360
        if delta_angles[i] > 180:
            delta_angles[i] = 360 - delta_angles[i]
    def angle_accuracy(d_angles, thres=10):
        pred_r = [dangle <= thres for dangle in d_angles]
        accu = sum(pred_r) / len(pred_r)
        return accu,
    accuracys = [angle_accuracy(delta_angles, thres) for thres in range(90)]
    print("num matched = ", len(delta_angles))
    print("thres = 10, accu = ", accuracys[10])
    print("thres = 20, accu = ", accuracys[20])
    print("thres = 30, accu = ", accuracys[30])

    
    return mAP


def color_white(image, mask, angle):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    h, w, _ = image.shape
    # green = 100 * np.ones((h,w,3)) + image

    alpha = 0.5
    color = [0,255,0]
    color_mask = np.zeros((h,w,3))
    color_mask[:,:,:] = color
    color_mask = image * (1-alpha) + alpha * color_mask
    # white = image * (1 - alpha) + alpha * np.ones((h,w,3)) * 255,
    # white = image        # rr, cc = skimage.draw.line(int(yc), int(xc), int(yc + 20 * math.sin(angle[i])), int(xc + 20*math.cos(angle[i])),width=10)
        # print("line",rr, cc)
        # image[rr, cc, :] = [255, 0, 0]
    # draw mask
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        one_mask = (np.sum(mask, -1, keepdims=True) >= 1)
        colored = np.where(one_mask, color_mask, image).astype(np.uint8)
        print("colored", colored.shape)
        # colored = colored[0, :, :,:]
    else:
        colored = gray.astype(np.uint8)

    # draw the line

    _, _, num = mask.shape
    print(mask.shape)
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


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set

    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)

    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "./results/splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
        
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "./results/splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect hands.')
    parser.add_argument("--command",
                        metavar="<command>",
                        help="'train' or 'splash'",
                        default="train")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/hand/dataset/",
                        help='Directory of the hand dataset',
                        default="./datasets/annotations/")
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'",
                        default="imagenet")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--imageset_path', required=False,
                        metavar="path to imageset",
                        help='imageset to apply the color splash effect on')
    parser.add_argument('--testset', required=False,
                        metavar="path to imageset",default="oxford",
                        help='imageset to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"
    elif args.command == "splash_set":
        assert args.imageset_path, "provide imageset_path"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = handConfig()
    else:
        class InferenceConfig(handConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
        try:
            model_file_name = "./model/hand_weights_{:%m%d%H%M}.h5".format(datetime.datetime.now())
        except:
            print("ERROR NAME")
            model_file_name = "./model/hand_weights_new.h5"

        model.keras_model.save_weights(model_file_name)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    elif args.command == "splash_set":
        file = open(args.imageset_path, "r")
        lines = file.readlines()
        for line in lines:
            line_split = line.split(",")
            image_path = line_split[0]
            print(image_path)
            detect_and_color_splash(model, image_path=image_path)
    elif args.command == "test":
        test(model, to_save=True)
    elif args.command == "test_bbox":
        test_bbox(model,to_save=True)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command)) 
