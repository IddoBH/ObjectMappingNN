import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from dataset_maker import get_dataset, make_tensors, MASKS, ROI_SIZE
from mapper import objectMapper

OUT_PATH = "/home/shovalg@staff.technion.ac.il/PycharmProjects/ObjectMappingNN/outputs/real"
OUT_NAME = "real"
DATASET_PATH_bb = "/home/shovalg@staff.technion.ac.il/PycharmProjects/ObjectMappingNN/real_images"


def test_bb(annotations, image_list, dataset_path, out_dir):
    for img_info in image_list:
        im = Image.open(os.path.join(dataset_path, img_info['file_name']))
        im_ann = list(filter(lambda im: im['image_id'] == img_info['id'], annotations))
        draw = ImageDraw.Draw(im)
        for ann in im_ann:
            bbox_ = ann['bbox']
            draw.rectangle((bbox_[0],bbox_[1],bbox_[0]+bbox_[2],bbox_[1]+bbox_[3]))
        im.save(os.path.join(out_dir, img_info['file_name']))


    # for im_id, centers in image_dict.items():
    #     # Create Image object
    #     im_info = list(filter(lambda im: im['id'] == im_id, image_list))[0]
    #     im = Image.open(os.path.join(dataset_path, im_info['file_name']))

        # # Draw
        # draw = ImageDraw.Draw(im)
        # for kp in centers:
        #     print(kp)
        #     draw.rectangle(kp[2])
        #
        # # Show image
        # im.save(os.path.join(out_dir, im_info['file_name']))


if __name__ == '__main__':
    print("getting dataset")
    test_image_list, test_annotations, test_categories = get_dataset(DATASET_PATH_bb, prt=True)
    test_bb(test_annotations, test_image_list, DATASET_PATH_bb, OUT_PATH)
    print("Done")
