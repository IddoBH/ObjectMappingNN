import json
import os

import numpy as np
import torch as torch
from PIL import Image, ImageOps

DATASET_PATH = "/Users/iddobar-haim/Library/CloudStorage/GoogleDrive-idodibh@gmail.com/My Drive/ARP/full_dataset"

ROI_SIZE = 32

MASKS = {
    "triangle"        : [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    "static_rectangle": [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    "static_ball"     : [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    "ceiling"         : [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    "floor"           : [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    "ball"            : [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    "rectangle"       : [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    "cart"            : [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    "pendulum"        : [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0.],
    "spring"          : [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.]
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
            tx1, tx2, tx3, ty1, ty2, ty3 = get_triangle_params(ann, bbox)
            targets.append([tx1, ty1, tx2, ty2, tx3, ty3, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            masks.append(MASKS["triangle"])
        elif ann['category_id'] == 2:
            tx1, tx2, tx3, tx4, ty1, ty2, ty3, ty4 = get_rectangle_params(ann, bbox)
            targets.append([0., 0., 0., 0., 0., 0., tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            masks.append(MASKS["static_rectangle"])
        elif ann['category_id'] == 3:
            tr, tx, ty = get_ball_params(ann, bbox)
            targets.append([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., tx, ty, tr, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            masks.append(MASKS["static_ball"])
        elif ann['category_id'] == 4:
            tx1, tx2, ty1, ty2 = get_line_params(ann, bbox)
            targets.append([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., tx1, ty1, tx2, ty2, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            masks.append(MASKS["ceiling"])
        elif ann['category_id'] == 5:
            tx1, tx2, ty1, ty2 = get_line_params(ann, bbox)
            targets.append([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., tx1, ty1, tx2, ty2, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            masks.append(MASKS["floor"])
        elif ann['category_id'] == 6:
            tr, tx, ty = get_ball_params(ann, bbox)
            targets.append([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., tx, ty, tr, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            masks.append(MASKS["ball"])
        elif ann['category_id'] == 7:
            tx1, tx2, tx3, tx4, ty1, ty2, ty3, ty4 = get_rectangle_params(ann, bbox)
            targets.append([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            masks.append(MASKS["rectangle"])
        elif ann['category_id'] == 8:
            tx1, tx2, tx3, tx4, ty1, ty2, ty3, ty4, tr, txc1, tyc1, txc2, tyc2 = get_cart_params(ann, bbox)
            targets.append([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4, txc1, tyc1, txc2, tyc2, tr, 0., 0., 0., 0., 0., 0., 0., 0.])
            masks.append(MASKS["cart"])
        elif ann['category_id'] == 9:
            tx1, tx2, ty1, ty2 = get_line_params(ann, bbox)
            targets.append([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., tx1, ty1, tx2, ty2, 0., 0., 0., 0.])
            masks.append(MASKS["pendulum"])
        elif ann['category_id'] == 10:
            tx1, tx2, ty1, ty2 = get_line_params(ann, bbox)
            targets.append(
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., tx1, ty1, tx2, ty2])
            masks.append(MASKS["spring"])
    return targets, masks


def get_line_params(ann, bbox):
    tx1 = point_transformation_x(bbox, ann['obj_params']['X_corner_1'])
    ty1 = point_transformation_y(bbox, ann['obj_params']['Y_corner_1'])
    tx2 = point_transformation_x(bbox, ann['obj_params']['X_corner_2'])
    ty2 = point_transformation_y(bbox, ann['obj_params']['Y_corner_2'])
    return tx1, tx2, ty1, ty2


def get_triangle_params(ann, bbox):
    tx1, tx2, ty1, ty2 = get_line_params(ann, bbox)
    tx3 = point_transformation_x(bbox, ann['obj_params']['X_corner_3'])
    ty3 = point_transformation_y(bbox, ann['obj_params']['Y_corner_3'])
    return tx1, tx2, tx3, ty1, ty2, ty3


def get_rectangle_params(ann, bbox):
    tx1, tx2, tx3, ty1, ty2, ty3 = get_triangle_params(ann, bbox)
    tx4 = point_transformation_x(bbox, ann['obj_params']['X_corner_4'])
    ty4 = point_transformation_y(bbox, ann['obj_params']['Y_corner_4'])
    return tx1, tx2, tx3, tx4, ty1, ty2, ty3, ty4


def get_cart_params(ann, bbox):
    tx1, tx2, tx3, tx4, ty1, ty2, ty3, ty4 = get_rectangle_params(ann, bbox)
    tr, txc1, tyc1 = get_ball_params(ann, bbox)
    txc2 = point_transformation_x(bbox, ann['obj_params']['X_center_2'])
    tyc2 = point_transformation_y(bbox, ann['obj_params']['Y_center_2'])
    return tx1, tx2, tx3, tx4, ty1, ty2, ty3, ty4, tr, txc1, tyc1, txc2, tyc2


def point_transformation_y(bbox, point):
    return (point - bbox[1]) * (ROI_SIZE / bbox[3]) if bbox[3] else point - bbox[1]


def point_transformation_x(bbox, point):
    return (point - bbox[0]) * (ROI_SIZE / bbox[2]) if bbox[2] else point - bbox[0]



def get_ball_params(ann, bbox):
    tx = point_transformation_x(bbox, ann['obj_params']['X_center'])
    ty = point_transformation_y(bbox, ann['obj_params']['Y_center'])
    tr = radius_transformation(bbox, ann['obj_params']['radius'])
    return tr, tx, ty


def radius_transformation(bbox, radius):
    return radius * (ROI_SIZE / bbox[2])


def crop_img(img, ann):
    bbox = ann['bbox']
    print(bbox)
    cropped = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    return np.asfarray(Image.fromarray(cropped).resize((ROI_SIZE, ROI_SIZE)))


def make_tensors(image_list, annotations, path=DATASET_PATH):
    X = []
    y = []
    mask = []
    for img_info in image_list:
        im_view = np.asarray(ImageOps.grayscale(Image.open(os.path.join(path, img_info['file_name']))))
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
