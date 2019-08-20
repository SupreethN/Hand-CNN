import math
import os
import numpy as np
import datetime
import skimage
import matplotlib.pyplot as plt
import cv2
from mrcnn import model as modellib, utils


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

    def load_orientations(self,image_id):
        """Generate instance orientaion for an image.
       Returns:
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "hand":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        orientations = np.zeros(shape=(len(info["polygons"]), 1), dtype=np.float32)
        for i, points in enumerate(info["polygons"]):
            x = [points[0], points[2], points[4], points[6]]
            y = [points[1], points[3], points[5], points[7]]
            x_wrist = (x[0] + x[1]) / 2
            y_wrist = (y[0] + y[1]) / 2
            x_hand = (x[2] + x[3]) / 2
            y_hand = (y[2] + y[3]) / 2
            x_delta = x_hand - x_wrist
            y_delta = y_hand - y_wrist
            orientations[i, 0] = math.atan2(y_delta, x_delta)
        return orientations

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "hand":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)




def get_rotate_rect(model, testset="oxford"):

    # Test dataset.
    dataset = "/home/weiuniverse/mask-rcnn/dataset/"
    dataset_test = HandDataset()
    if testset == "tv":
        print("loading test_IDs.txt")
        dataset_test.load_hand(dataset + "test_IDs.txt")
    elif testset == "oxford":
        print("loading oxford_annos.txt")
        dataset_test.load_hand(dataset + "oxford_annos.txt")
    else:
        print("No DataSet")
    dataset_test.prepare()

    dir_name = "./samples/hand/results/{:%m%d%H%M}/".format(datetime.datetime.now())
    os.mkdir(dir_name)

    rotate_rect_name = dir_name + "pred_rotate_rect.txt".format(datetime.datetime.now())
    rotate_rect_file = open(rotate_rect_name, "w")

    for image_info in dataset_test.image_info:
        print(image_info)
        image_id = image_info['id']
        image_path = image_info['path']
        img = skimage.io.imread(image_path)
        result = model.detect([img], verbose=0)[0]
        pred_scores = result["scores"]
        pred_bboxes = result["rois"]
        pred_masks = result["masks"]

        # generate the min_rects
        pred_min_rects = np.zeros(shape=(pred_masks.shape[-1], 8))
        for i in range(pred_masks.shape[-1]):
            mask_points = np.transpose(np.nonzero(pred_masks[:,:,i]))
            # (xc, yc), (h, w), angle
            bx1, by1, bw, bh = cv2.boundingRect(points=mask_points)
            bx2 = bx1 + bw
            by2 = by1 + bh
            bbox = pred_bboxes[i]
            rect = cv2.minAreaRect(points=mask_points)
            box = cv2.boxPoints(rect)
            x = box[:, 1]
            y = box[:, 0]
            x = [int(tmp) for tmp in x]
            y = [int(tmp) for tmp in y]
            for inx in range(4):
                if x[inx] >= image_info["width"]:
                    x[inx] = image_info["width"] - 1
                elif x[inx] <= 0:
                    x[inx] = 0
                if y[inx] >= image_info["height"]:
                    y[inx] = image_info["height"] - 1
                elif y[inx] <= 0:
                    y[inx] = 0
            box = [x[0], y[0], x[1], y[1], x[2], y[2], x[3], y[3]]
            pred_min_rects[i] = box
            rotate_rect_file.write(image_path+','+
                            str(bbox[1])+','+str(bbox[0])+','+
                            str(bbox[3])+','+str(bbox[2])+','+
                            str(x[0])+','+str(y[0])+','+
                            str(x[1])+','+str(y[1])+','+
                            str(x[2])+','+str(y[2])+','+
                            str(x[3])+','+str(y[3])+','+str(pred_scores[i])+"\n")
            rr, cc = skimage.draw.polygon(y, x)
            pred_masks[rr, cc, i] = 1
    rotate_rect_file.close()


