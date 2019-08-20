from hand import HandDataset
import skimage

class HandDataset_count(HandDataset):
    def count_mask_pixels(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "hand":
            return super(self.__class__, self).load_mask(image_id)
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        Part = []
        for i, points in enumerate(info["polygons"]):
            x = [points[0], points[2], points[4], points[6]]
            y = [points[1], points[3], points[5], points[7]]
            for inx in range(4):
                if x[inx] >= info["width"]:
                    x[inx] = info["width"] - 1
                if y[inx] >= info["height"]:
                    y[inx] = info["height"] - 1
            rr, cc = skimage.draw.polygon(y, x)
            ratio = rr.shape[0]
            # if ratio < 100:
            #     print(ratio)
            # if ratio < 0.000352:
            #     print(info["path"], points, ratio)
            Part.append(ratio)
        return Part
            # print(cc.shape)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # return len(rr)


dataset_test = HandDataset_count()
# dataset_test.load_hand("/home/weiuniverse/mask-rcnn/dataset/" + "oxford_annos.txt")
dataset_test.load_hand("/home/weiuniverse/mask-rcnn/dataset/" + "train_IDs_tv.txt")
dataset_test.load_hand("/home/weiuniverse/mask-rcnn/dataset/" + "valid_IDs.txt")
dataset_test.load_hand("/home/weiuniverse/mask-rcnn/dataset/" + "test_IDs_hand.txt")

img_info = dataset_test.image_info

Part = []
for data in img_info:
    P = dataset_test.count_mask_pixels(data["id"])
    Part.extend(P)

import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(color_codes=true)
# sns.hist(Part)
plt.hist(Part,range=[0,20000],bins=10)
plt.xlabel("Hand Size")
plt.ylabel("count")
plt.title("Histogram of Hand Size")
plt.savefig("hist_hand.pdf")

import pickle

with open("hand_stats.pkl","wb") as file:
    pickle.dump(obj=Part, file=file)
# print(min(Part))
