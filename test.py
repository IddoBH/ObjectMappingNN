import os

import torch
from PIL import Image, ImageDraw
from dataset_maker import DATASET_PATH, get_dataset, make_tensors, MASKS
from mapper import objectMapper

OUT_PATH = "/Users/iddobar-haim/PycharmProjects/ObjectMappingNN/outputs"
OUT_NAME = "cat"
MODEL_PATH = "/Users/iddobar-haim/PycharmProjects/ObjectMappingNN/models/circles_and_triangles.pth"


def test_loop(model, X, annotations, image_list):
    predictions = model.forward(X)
    image_dict = {}
    for pred, ann in zip(predictions, annotations):
        bbox = ann['bbox']
        if ann['image_id'] not in image_dict:
            image_dict[ann['image_id']] = []
        if ann['category_id'] == 1:
            add_triangle_test(ann, bbox, pred[0:6], image_dict)
        elif ann['category_id'] == 3:
            add_ball_test(ann, bbox, pred[6:9], image_dict)
        elif ann['category_id'] == 6:
            add_ball_test(ann, bbox, pred[9:12], image_dict)

    out_dir = os.path.join(OUT_PATH, OUT_NAME)
    os.mkdir(out_dir)
    print(image_dict)

    for im_id, centers in image_dict.items():
        # Create Image object
        im_info = list(filter(lambda im: im['id'] == im_id, image_list))[0]
        im = Image.open(os.path.join(DATASET_PATH, 'val', im_info['file_name']))

        #Draw
        draw = ImageDraw.Draw(im)
        for kp in centers:
            if kp[0] == "ball":
                draw.point(kp[1][0:2], fill='red')
                draw.line([kp[1][0], kp[1][1], kp[1][0] + kp[1][2], kp[1][1]])
            elif kp[0] == "triangle":
                draw.point(kp[1], fill='red')

        # Show image
        im.save(os.path.join(out_dir, im_info['file_name']))


def add_triangle_test(ann, bbox, center, image_dict):
    tx1, tx2, tx3, ty1, ty2, ty3 = get_triangle_params_test(bbox, center)
    image_dict[ann['image_id']].append(("triangle", (tx1, ty1, tx2, ty2, tx3, ty3)))


def get_triangle_params_test(bbox, center):
    tx1 = round(center[0].item() * (bbox[2] / 32) + bbox[0])
    ty1 = round(center[1].item() * (bbox[3] / 32) + bbox[1])
    tx2 = round(center[2].item() * (bbox[2] / 32) + bbox[0])
    ty2 = round(center[3].item() * (bbox[3] / 32) + bbox[1])
    tx3 = round(center[4].item() * (bbox[2] / 32) + bbox[0])
    ty3 = round(center[5].item() * (bbox[3] / 32) + bbox[1])
    return tx1, tx2, tx3, ty1, ty2, ty3


def add_ball_test(ann, bbox, center, image_dict):
    tr, tx, ty = get_ball_params_test(bbox, center)
    image_dict[ann['image_id']].append(("ball", (tx, ty, tr)))


def get_ball_params_test(bbox, center):
    tx = round(center[0].item() * (bbox[2] / 32) + bbox[0])
    ty = round(center[1].item() * (bbox[3] / 32) + bbox[1])
    tr = round(center[2].item() * (bbox[2] + bbox[3]) / 64)
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