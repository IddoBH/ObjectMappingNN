import json
import os

import numpy as np
import torch as torch
from PIL import Image, ImageOps

DATASET_PATH_test = "/home/shovalg@staff.technion.ac.il/PycharmProjects/ObjectMappingNN/real_images"
DATASET_PATH_train = "/home/shovalg@staff.technion.ac.il/PycharmProjects/ObjectMappingNN/real_images"
DATASET_PATH_test_gen = "/home/shovalg@staff.technion.ac.il/PycharmProjects/ObjectMappingNN/iddos_drive/full_dataset"
DATASET_PATH_train_gen = "/home/shovalg@staff.technion.ac.il/PycharmProjects/ObjectMappingNN/iddos_drive/full_dataset"

ROI_SIZE = 32

MASKS = {
    "triangle": [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0.],
    "static_rectangle": [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0., 0., 0., 0.],
    "static_ball": [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0.],
    "ceiling": [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.],
    "floor": [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0.,
              0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.],
    "ball": [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
             1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0.],
    "rectangle": [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0.],
    "cart": [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.,
             0., 0., 0.],
    "pendulum": [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.,
                 1., 0., 0., 0., 0.],
    "spring": [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 1., 1., 1., 1.]
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


def make_target_tensor(annotations, transpose=False):
    targets = []
    masks = []
    for ann in annotations:
        bbox = ann['bbox']
        if ann['category_id'] == 1:
            tx1, tx2, tx3, ty1, ty2, ty3 = get_triangle_params(ann, bbox, transpose)
            dx12, dy12 = tx1 - tx2, ty1 - ty2
            dx13, dy13 = tx1 - tx3, ty1 - ty3
            dx23, dy23 = tx3 - tx2, ty3 - ty2
            d12 = dx12*dx12 + dy12*dy12
            d13 = dx13 * dx13 + dy13 * dy13
            d23 = dx23 * dx23 + dy23 * dy23
            if d12 > d13 and d12 > d23:
                tx2, tx3 = tx3, tx2
                ty2, ty3 = ty3, ty2
            elif d23 > d13 and d23 > d12:
                tx1, tx2 = tx2, tx1
                ty1, ty2 = ty2, ty1

            targets.append(
                [tx1, ty1, tx2, ty2, tx3, ty3, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0.])
            masks.append(MASKS["triangle"])
        elif ann['category_id'] == 2:
            tx1, tx2, tx3, tx4, ty1, ty2, ty3, ty4 = get_rectangle_params(ann, bbox, transpose)
            targets.append(
                [0., 0., 0., 0., 0., 0., tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0.])
            masks.append(MASKS["static_rectangle"])
        elif ann['category_id'] == 3:
            tr, tx, ty = get_ball_params(ann, bbox, transpose)
            targets.append(
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., tx, ty, tr, 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0.])
            masks.append(MASKS["static_ball"])
        elif ann['category_id'] == 4:
            if transpose:
                continue
            tx1, tx2, ty1, ty2 = get_line_params(ann, bbox, transpose)
            targets.append(
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., tx1, ty1, tx2, ty2, 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0.])
            masks.append(MASKS["ceiling"])
        elif ann['category_id'] == 5:
            if transpose:
                continue
            tx1, tx2, ty1, ty2 = get_line_params(ann, bbox, transpose)
            targets.append(
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., tx1, ty1, tx2, ty2,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0.])
            masks.append(MASKS["floor"])
        elif ann['category_id'] == 6:
            tr, tx, ty = get_ball_params(ann, bbox, transpose)
            targets.append(
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., tx,
                 ty, tr, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0.])
            masks.append(MASKS["ball"])
        elif ann['category_id'] == 7:
            tx1, tx2, tx3, tx4, ty1, ty2, ty3, ty4 = get_rectangle_params(ann, bbox, transpose)
            targets.append(
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0.])
            masks.append(MASKS["rectangle"])
        elif ann['category_id'] == 8:
            if transpose:
                continue
            tx1, tx2, tx3, tx4, ty1, ty2, ty3, ty4, tr, txc1, tyc1, txc2, tyc2 = get_cart_params(ann, bbox, transpose)
            targets.append(
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4, txc1, tyc1, txc2, tyc2,
                 tr, 0., 0., 0., 0., 0., 0., 0., 0.])
            masks.append(MASKS["cart"])
        elif ann['category_id'] == 9:
            tx1, tx2, ty1, ty2 = get_line_params(ann, bbox, transpose)
            targets.append(
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., tx1, ty1,
                 tx2, ty2, 0., 0., 0., 0.])
            masks.append(MASKS["pendulum"])
        elif ann['category_id'] == 10:
            tx1, tx2, ty1, ty2 = get_line_params(ann, bbox, transpose)
            targets.append(
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., tx1, ty1, tx2, ty2])
            masks.append(MASKS["spring"])
    return targets, masks


def sort_corners(corners):
    return tuple(sorted(corners))


def get_line_params(ann, bbox, transpose):
    tx1 = point_transformation_x(bbox, ann['obj_params']['X_corner_1'])
    ty1 = point_transformation_y(bbox, ann['obj_params']['Y_corner_1'])
    tx2 = point_transformation_x(bbox, ann['obj_params']['X_corner_2'])
    ty2 = point_transformation_y(bbox, ann['obj_params']['Y_corner_2'])
    if transpose:
        tx1, ty1 = ty1, tx1
        tx2, ty2 = ty2, tx2
    sc = sort_corners(((tx1, ty1), (tx2, ty2)))
    return sc[0][0], sc[1][0], sc[0][1], sc[1][1]  # tx1, tx2, ty1, ty2


def get_triangle_params(ann, bbox, transpose):
    tx1, tx2, ty1, ty2 = get_line_params(ann, bbox, transpose)
    tx3 = point_transformation_x(bbox, ann['obj_params']['X_corner_3'])
    ty3 = point_transformation_y(bbox, ann['obj_params']['Y_corner_3'])
    if transpose:
        tx3, ty3 = ty3, tx3
    sc = sort_corners(((tx1, ty1), (tx2, ty2), (tx3, ty3)))
    return sc[0][0], sc[1][0], sc[2][0], sc[0][1], sc[1][1], sc[2][1]  # tx1, tx2, tx3, ty1, ty2, ty3


def get_rectangle_params(ann, bbox, transpose):
    tx1, tx2, tx3, ty1, ty2, ty3 = get_triangle_params(ann, bbox, transpose)
    tx4 = point_transformation_x(bbox, ann['obj_params']['X_corner_4'])
    ty4 = point_transformation_y(bbox, ann['obj_params']['Y_corner_4'])
    if transpose:
        tx4, ty4 = ty4, tx4
    sc = sort_corners(((tx1, ty1), (tx2, ty2), (tx3, ty3), (tx4, ty4)))
    return sc[0][0], sc[1][0], sc[2][0], sc[3][0], sc[0][1], sc[1][1], sc[2][1], sc[3][
        1]  # tx1, tx2, tx3, tx4, ty1, ty2, ty3, ty4


def get_cart_params(ann, bbox, transpose):
    tx1, tx2, tx3, tx4, ty1, ty2, ty3, ty4 = get_rectangle_params(ann, bbox, transpose)
    tr, txc1, tyc1 = get_ball_params(ann, bbox, transpose)
    txc2 = point_transformation_x(bbox, ann['obj_params']['X_center_2'])
    tyc2 = point_transformation_y(bbox, ann['obj_params']['Y_center_2'])
    if transpose:
        txc2, tyc2 = tyc2, txc2
    return tx1, tx2, tx3, tx4, ty1, ty2, ty3, ty4, tr, txc1, tyc1, txc2, tyc2


def point_transformation_y(bbox, point):
    return (point - bbox[1]) * (ROI_SIZE / bbox[3]) if bbox[3] else point - bbox[1]


def point_transformation_x(bbox, point):
    return (point - bbox[0]) * (ROI_SIZE / bbox[2]) if bbox[2] else point - bbox[0]


def get_ball_params(ann, bbox, transpose):
    tx = point_transformation_x(bbox, ann['obj_params']['X_center'])
    ty = point_transformation_y(bbox, ann['obj_params']['Y_center'])
    tr = radius_transformation(bbox, ann['obj_params']['radius'])
    if transpose:
        tx, ty = ty, tx
    return tr, tx, ty


def radius_transformation(bbox, radius):
    return radius * (ROI_SIZE / bbox[2])


def crop_img(img, ann):
    bbox = tuple(map(round, ann['bbox']))
    print(bbox)
    cropped = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    return np.asfarray(Image.fromarray(cropped).resize((ROI_SIZE, ROI_SIZE)))


def make_X(image_list, annotations, path):
    X = []
    for img_info in image_list:
        im_view = np.asarray(ImageOps.grayscale(Image.open(os.path.join(path, img_info['file_name']))))
        im_ann = list(filter(lambda im: im['image_id'] == img_info['id'], annotations))
        for ia in im_ann:
            ci = crop_img(im_view, ia)
            X.append(ci.flatten())
    X = torch.tensor(np.asarray(X), dtype=torch.float32, requires_grad=True)
    return X


def make_tensors(image_list, annotations, path):
    X = []
    y = []
    mask = []

    for img_info in image_list:
        im_view = np.asarray(ImageOps.grayscale(Image.open(os.path.join(path, img_info['file_name']))))
        im_ann = list(filter(lambda im: im['image_id'] == img_info['id'], annotations))
        insert_once(X, y, mask, im_view, im_ann)
        insert_once(X, y, mask, im_view, im_ann, transpose=True)

    X = torch.tensor(np.asarray(X), dtype=torch.float32, requires_grad=True)
    y = torch.tensor(np.asarray(y), dtype=torch.float32, requires_grad=True)
    mask = torch.tensor(np.asarray(mask), dtype=torch.float32, requires_grad=True)
    if True:
        print(X, X.shape)
        print(y, y.shape)
        print(mask, mask.shape)
    return X, y, mask


def tensors(dataset):
    print("getting data")
    train_image_list, train_annotations, train_categories = get_dataset(dataset)
    print("making tensors")
    return make_tensors(train_image_list, train_annotations, dataset)


def make_tv():
    x_1, y_1, mask_1 = tensors(
        "/home/shovalg@staff.technion.ac.il/PycharmProjects/ObjectMappingNN/iddos_drive/full_dataset/train/set_1")
    x_2, y_2, mask_2 = tensors(
        "/home/shovalg@staff.technion.ac.il/PycharmProjects/ObjectMappingNN/iddos_drive/full_dataset/train/set_2")
    x_3, y_3, mask_3 = tensors(
        "/home/shovalg@staff.technion.ac.il/PycharmProjects/ObjectMappingNN/iddos_drive/full_dataset/train/set_3")
    real_img_path = "/home/shovalg@staff.technion.ac.il/PycharmProjects/ObjectMappingNN/real_images"
    train_image_list, train_annotations, train_categories = get_dataset(real_img_path)
    X_train = []
    y_train = []
    mask_train = []

    X_val = []
    y_val = []
    mask_val = []

    val_names = []

    idx = 0
    for img_info in train_image_list:
        if "our" in img_info['file_name']:
            continue
        if idx % 5:
            insert_annotations(X_train, y_train, mask_train, img_info, train_annotations, real_img_path)
        else:
            val_names.append(img_info['file_name'])
            insert_annotations(X_val, y_val, mask_val, img_info, train_annotations, real_img_path)
        idx += 1
    print(val_names)

    X_train = torch.tensor(np.asarray(X_train), dtype=torch.float32, requires_grad=True)
    y_train = torch.tensor(np.asarray(y_train), dtype=torch.float32, requires_grad=True)
    mask_train = torch.tensor(np.asarray(mask_train), dtype=torch.float32, requires_grad=True)
    X_val = torch.tensor(np.asarray(X_val), dtype=torch.float32, requires_grad=True)
    y_val = torch.tensor(np.asarray(y_val), dtype=torch.float32, requires_grad=True)
    mask_val = torch.tensor(np.asarray(mask_val), dtype=torch.float32, requires_grad=True)

    X_train = torch.cat((x_1, x_2, x_3, X_train), 0)
    y_train = torch.cat((y_1, y_2, y_3, y_train), 0)
    mask_train = torch.cat((mask_1, mask_2, mask_3, mask_train), 0)

    return X_train, y_train, mask_train, X_val, y_val, mask_val


def insert_annotations(X, y, mask, img_info, train_annotations, real_img_path):
    im_view = np.asarray(ImageOps.grayscale(Image.open(os.path.join(real_img_path, img_info['file_name']))))
    im_ann = list(filter(lambda im: im['image_id'] == img_info['id'], train_annotations))
    insert_once(X, y, mask, im_view, im_ann)
    insert_once(X, y, mask, im_view, im_ann, transpose=True)


def insert_once(X, y, mask, im_view, im_ann, transpose=False):
    one_image_y, one_img_mask = make_target_tensor(im_ann, transpose)
    y.extend(one_image_y)
    mask.extend(one_img_mask)
    for ia in im_ann:
        ci = crop_img(im_view, ia)
        if transpose:
            if ia['category_id'] in (4, 5, 8):
                continue
            ci = ci.transpose()
        X.append(ci.flatten())


if __name__ == '__main__':
    make_tv()
