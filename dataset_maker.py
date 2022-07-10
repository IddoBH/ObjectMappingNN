import json
import os

import numpy as np
import torch as torch
from PIL import ImageDraw, ImageOps, Image

DATASET_PATH = "/Users/iddobar-haim/Library/CloudStorage/GoogleDrive-idodibh@gmail.com/My Drive/ARP/circles"


ROI_SIZE = 32

MASKS = {
    "static_ball": [1., 1., 0., 0.],
    "ball"       : [0., 0., 1., 1.]
}


def get_dataset(dir_path, prt=False):
    with open(os.path.join(dir_path, "COCO.json")) as ann:
        data = json.load(ann)
    image_list = data['images']
    annotations = data['annotations']
    categories = data['categories']
    if prt:
        print(data, '\n')
        print(image_list, '\n')
        print(annotations, '\n')
        print(categories)
    return image_list, annotations, categories


def make_target_tensor(annotations):
    targets = []
    masks = []
    for ann in annotations:
        bbox = ann['bbox']
        tx = (ann['obj_params']['x1'] - bbox[0]) * (32 / bbox[2])
        ty = (ann['obj_params']['y1'] - bbox[1]) * (32 / bbox[3])
        if ann['category_id'] == 3:
            targets.append([tx, ty, 0., 0.])
            masks.append(MASKS["static_ball"])
        elif ann['category_id'] == 6:
            targets.append([0., 0., tx, ty])
            masks.append(MASKS["ball"])
    return targets, masks


def crop_img(img, ann):
    bbox = ann['bbox']
    cropped = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    return np.asfarray(Image.fromarray(cropped).resize((ROI_SIZE, ROI_SIZE)))


def make_tensors(image_list, annotations, prt=False):
    X = []
    y = []
    mask = []
    for img_info in image_list:
        im_view = np.asarray(ImageOps.grayscale(Image.open(os.path.join(DATASET_PATH, "train", img_info['file_name']))))
        im_ann = list(filter(lambda im: im['image_id'] == img_info['id'], annotations))
        one_image_y, one_img_mask = make_target_tensor(im_ann)
        y.extend(one_image_y)
        mask.extend(one_img_mask)
        for ia in im_ann:
            ci = crop_img(im_view, ia)
            X.append(ci.flatten())

    X = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    y = torch.tensor(y, dtype=torch.float32, requires_grad=True)
    mask = torch.tensor(mask, dtype=torch.float32, requires_grad=True)
    if prt:
        print(X, X.shape)
        print(y, y.shape)
        print(mask, mask.shape)
    return X, y, mask
