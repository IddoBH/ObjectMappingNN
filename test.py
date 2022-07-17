import os

import torch
from PIL import Image, ImageDraw
from dataset_maker import DATASET_PATH, get_dataset, make_tensors, MASKS, ROI_SIZE
from mapper import objectMapper

OUT_PATH = "/Users/iddobar-haim/PycharmProjects/ObjectMappingNN/outputs"
OUT_NAME = "trio"
MODEL_PATH = "/Users/iddobar-haim/PycharmProjects/ObjectMappingNN/models/triangles.pth"


def test_loop(model, X, annotations, image_list, dataset_path=os.path.join(DATASET_PATH, 'val'), out_dir=os.path.join(OUT_PATH, OUT_NAME)):
    predictions = model.forward(X) if model else X
    image_dict = prepare_image_dict(annotations, predictions)

    # os.mkdir(out_dir)
    # print(image_dict)

    for im_id, centers in image_dict.items():
        # Create Image object
        im_info = list(filter(lambda im: im['id'] == im_id, image_list))[0]
        im = Image.open(os.path.join(dataset_path, im_info['file_name']))

        # Draw
        draw = ImageDraw.Draw(im)
        for kp in centers:
            print(kp)
            if kp[0] == "ball":
                draw.point(kp[1][0:2], fill='red')
                draw.line([kp[1][0], kp[1][1], kp[1][0] + kp[1][2], kp[1][1]])
            elif kp[0] in ("triangle", "rectangle", "line"):
                draw.point(kp[1], fill='red')
            elif kp[0] == "cart":
                draw.point(kp[1][0:12], fill='red')

                draw.line([kp[1][8], kp[1][9], kp[1][8] + kp[1][12], kp[1][9]])
                draw.line([kp[1][10], kp[1][11], kp[1][10] + kp[1][12], kp[1][11]])

        # Show image
        im.save(os.path.join(out_dir, im_info['file_name']))


def prepare_image_dict(annotations, predictions):
    image_dict = {}
    for pred, ann in zip(predictions, annotations):
        bbox = ann['bbox']
        if ann['image_id'] not in image_dict:
            image_dict[ann['image_id']] = []
        if ann['category_id'] == 1:
            add_triangle_test(ann, bbox, get_pred_slice(pred, "triangle"), image_dict)
        elif ann['category_id'] == 2:
            add_rectangle_test(ann, bbox, get_pred_slice(pred, "static_rectangle"), image_dict)
        elif ann['category_id'] == 3:
            add_ball_test(ann, bbox, get_pred_slice(pred, "static_ball"), image_dict)
        elif ann['category_id'] == 4:
            add_line_test(ann, bbox, get_pred_slice(pred, "ceiling"), image_dict)
        elif ann['category_id'] == 5:
            add_line_test(ann, bbox, get_pred_slice(pred, "floor"), image_dict)
        elif ann['category_id'] == 6:
            add_ball_test(ann, bbox, get_pred_slice(pred, "ball"), image_dict)
        elif ann['category_id'] == 7:
            add_rectangle_test(ann, bbox, get_pred_slice(pred, "rectangle"), image_dict)
        elif ann['category_id'] == 8:
            add_cart_test(ann, bbox, get_pred_slice(pred, "cart"), image_dict)
        elif ann['category_id'] == 9:
            add_line_test(ann, bbox, get_pred_slice(pred, "pendulum"), image_dict)
        elif ann['category_id'] == 10:
            pass
    return image_dict


def get_pred_slice(pred, shape_name):
    start = MASKS[shape_name].index(1.)
    length = int(sum(MASKS[shape_name]))
    pred_slice = pred[start:start + length]
    return pred_slice


def add_line_test(ann, bbox, center, image_dict):
    tx1, tx2, ty1, ty2 = get_line_params_test(bbox, center)
    image_dict[ann['image_id']].append(("line", (tx1, ty1, tx2, ty2)))


def get_line_params_test(bbox, center):
    tx1 = round(center[0].item() * (bbox[2] / ROI_SIZE) + bbox[0])
    ty1 = round(center[1].item() * (bbox[3] / ROI_SIZE) + bbox[1])
    tx2 = round(center[2].item() * (bbox[2] / ROI_SIZE) + bbox[0])
    ty2 = round(center[3].item() * (bbox[3] / ROI_SIZE) + bbox[1])
    return tx1, tx2, ty1, ty2


def add_triangle_test(ann, bbox, center, image_dict):
    tx1, tx2, tx3, ty1, ty2, ty3 = get_triangle_params_test(bbox, center)
    image_dict[ann['image_id']].append(("triangle", (tx1, ty1, tx2, ty2, tx3, ty3)))


def get_triangle_params_test(bbox, center):
    tx1, tx2, ty1, ty2 = get_line_params_test(bbox, center)
    tx3 = round(center[4].item() * (bbox[2] / ROI_SIZE) + bbox[0])
    ty3 = round(center[5].item() * (bbox[3] / ROI_SIZE) + bbox[1])
    return tx1, tx2, tx3, ty1, ty2, ty3


def add_rectangle_test(ann, bbox, center, image_dict):
    tx1, tx2, tx3, tx4, ty1, ty2, ty3, ty4 = get_rectangle_params_test(bbox, center)
    image_dict[ann['image_id']].append(("rectangle", (tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4)))


def get_rectangle_params_test(bbox, center):
    tx1, tx2, tx3, ty1, ty2, ty3 = get_triangle_params_test(bbox, center)
    tx4 = round(center[6].item() * (bbox[2] / ROI_SIZE) + bbox[0])
    ty4 = round(center[7].item() * (bbox[3] / ROI_SIZE) + bbox[1])
    return tx1, tx2, tx3, tx4, ty1, ty2, ty3, ty4


def add_cart_test(ann, bbox, center, image_dict):
    tx1, tx2, tx3, tx4, ty1, ty2, ty3, ty4, txc1, tyc1, txc2, tyc2, tr = get_cart_params_test(bbox, center)
    image_dict[ann['image_id']].append(("cart", (tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4, txc1, tyc1, txc2, tyc2, tr)))


def get_cart_params_test(bbox, center):
    tx1, tx2, tx3, tx4, ty1, ty2, ty3, ty4 = get_rectangle_params_test(bbox, center)
    tr, txc1, tyc1 = get_ball_params_test(bbox, center, start=8)
    tr, txc2, tyc2 = get_ball_params_test(bbox, center, start=10)
    return tx1, tx2, tx3, tx4, ty1, ty2, ty3, ty4, txc1, tyc1, txc2, tyc2, tr


def add_ball_test(ann, bbox, center, image_dict):
    tr, tx, ty = get_ball_params_test(bbox, center)
    image_dict[ann['image_id']].append(("ball", (tx, ty, tr)))


def get_ball_params_test(bbox, center, start=0):
    tx = round(center[start].item() * (bbox[2] / ROI_SIZE) + bbox[0])
    ty = round(center[start + 1].item() * (bbox[3] / ROI_SIZE) + bbox[1])
    tr = round(center[start + 2].item() * (bbox[2] + bbox[3]) / 64)
    return tr, tx, ty


if __name__ == '__main__':
    print("making nn")
    model = objectMapper(len(MASKS["ball"]))
    model.load_state_dict(torch.load(MODEL_PATH))
    print("getting dataset")
    test_image_list, test_annotations, test_categories = get_dataset(os.path.join(DATASET_PATH, 'val'), prt=True)
    test_X, test_y, test_mask = make_tensors(test_image_list, test_annotations)
    print("testing")
    test_loop(model, test_X, test_annotations, test_image_list)
    print("Done")
