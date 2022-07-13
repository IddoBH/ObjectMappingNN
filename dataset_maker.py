import json
import os

import numpy as np
import torch as torch
from PIL import ImageDraw, ImageOps, Image

DATASET_PATH = "/Users/iddobar-haim/Library/CloudStorage/GoogleDrive-idodibh@gmail.com/My Drive/ARP/triangles"


ROI_SIZE = 32

MASKS = {
    "triangle"   : [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
    "static_ball": [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
    "ball"       : [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.]
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
        if ann['category_id'] == 1:
            tx1 = point_transformation_x(bbox, ann['obj_params']['X_corner_1'])
            ty1 = point_transformation_y(bbox, ann['obj_params']['Y_corner_1'])
            tx2 = point_transformation_x(bbox, ann['obj_params']['X_corner_2'])
            ty2 = point_transformation_y(bbox, ann['obj_params']['Y_corner_2'])
            tx3 = point_transformation_x(bbox, ann['obj_params']['X_corner_3'])
            ty3 = point_transformation_y(bbox, ann['obj_params']['Y_corner_3'])
            targets.append([tx1, ty1, tx2, ty2, tx3, ty3, 0., 0., 0., 0., 0., 0.])
            masks.append(MASKS["triangle"])
        elif ann['category_id'] == 3:
            tr, tx, ty = get_ball_params(ann, bbox)
            targets.append([0., 0., 0., 0., 0., 0., tx, ty, tr, 0., 0., 0.])
            masks.append(MASKS["static_ball"])
        elif ann['category_id'] == 6:
            tr, tx, ty = get_ball_params(ann, bbox)
            targets.append([0., 0., 0., 0., 0., 0., 0., 0., 0., tx, ty, tr])
            masks.append(MASKS["ball"])
    return targets, masks


def point_transformation_y(bbox, point):
    return (point - bbox[1]) * (32 / bbox[3])


def point_transformation_x(bbox, point):
    return (point - bbox[0]) * (32 / bbox[2])


def get_ball_params(ann, bbox):
    tx = point_transformation_x(bbox, ann['obj_params']['X_center'])
    ty = point_transformation_y(bbox, ann['obj_params']['Y_center'])
    tr = radius_transformation(bbox, ann['obj_params']['radius'])
    return tr, tx, ty


def radius_transformation(bbox, radius):
    return radius * (64 / (bbox[2] + bbox[3]))


def crop_img(img, ann):
    bbox = ann['bbox']
    cropped = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    return np.asfarray(Image.fromarray(cropped).resize((ROI_SIZE, ROI_SIZE)))


def make_tensors(image_list, annotations):
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
    if True:
        print(X, X.shape)
        print(y, y.shape)
        print(mask, mask.shape)
    return X, y, mask
