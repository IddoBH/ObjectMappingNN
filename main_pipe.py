import json
import os.path

import numpy as np
import torch
from PIL import ImageOps, Image
from numpy.linalg import norm

from dataset_maker import get_dataset, MASKS, crop_img
from mapper import objectMapper
from test import get_pred_slice, get_triangle_params_test, get_cart_params_test, get_line_params_test, \
    get_ball_params_test, get_rectangle_params_test


def main(model_path, image_path, output_path):
    model = objectMapper(len(MASKS["ball"]))
    model.load_state_dict(torch.load(model_path))

    im_ann, im_view = prepare_annotations(image_path)

    predictions = predict(im_ann, im_view, model)

    output_dict = make_output_obj()

    fill_output_obj(im_ann, output_dict, predictions)

    dump_output_to_json(output_dict, output_path)


def fill_output_obj(im_ann, output_dict, predictions):
    connectors = []
    for pred, ann in zip(predictions, im_ann):
        bbox = ann['bbox']
        if ann['category_id'] == 1:
            prepare_triangle(bbox, output_dict, pred)
        elif ann['category_id'] in (2, 7):
            prepare_block(bbox, output_dict, pred, ann['category_id'])
        elif ann['category_id'] in (3, 6):
            prepare_ball(bbox, output_dict, pred, ann['category_id'])
        elif ann['category_id'] in (4, 5):
            prepare_wall(bbox, output_dict, pred, ann['category_id'])
        elif ann['category_id'] == 8:
            prepare_cart(bbox, output_dict, pred)
        else:
            connectors.append((bbox, pred, ann['category_id']))

    for con in connectors:
        prepare_connector(con[0], output_dict, con[1], con[2])


def prepare_block(bbox, output_dict, pred, category_id):
    cat = 'static_rectangle' if category_id == 2 else 'rectangle'
    tx1, tx2, tx3, tx4, ty1, ty2, ty3, ty4 = get_rectangle_params_test(bbox, get_pred_slice(pred, cat))
    output_dict['Blocks'].append({
        "A": xy_to_str(tx1, ty1),
        "B": xy_to_str(tx2, ty2),
        "C": xy_to_str(tx3, ty3),
        "D": xy_to_str(tx4, ty4),
        "IsStatic": category_id == 2
    })


def prepare_connector(bbox, output_dict, pred, category_id):
    cat = 'pendulum' if category_id == 9 else 'spring'
    tx1, tx2, ty1, ty2 = get_line_params_test(bbox, get_pred_slice(pred, cat))
    entry = {}
    for side, p in zip(('A', 'B'), ((tx1, ty1), (tx2, ty2))):
        current_best = ('', 0, 1000000)
        for category, shapes in output_dict.items():
            if category in ("Lines", "Springs"):
                continue

            for idx, shape in enumerate(shapes):
                if category == "Balls":
                    current_best = measure_distance_to_ball(category, current_best, idx, p, shape)
                if category == "Walls":
                    current_best = measure_distance_to_wall(category, current_best, idx, p, shape)
                elif category in ("Blocks", "Carts", "Triangles"):
                    current_best = measure_distance_to_polygon(category, current_best, idx, p, shape)


            entry[f'connection{side}'] = current_best[0]
            entry[f'index{side}'] = current_best[1]

    cat2 = 'Lines' if category_id == 9 else 'Springs'
    output_dict[cat2].append(entry)


def measure_distance_to_polygon(category, current_best, idx, p, shape):
    points = []
    for key, point in shape.items():
        if key not in ('A', 'B', 'C', 'D'):
            continue
        points.append(str_to_point(point))
    cm = np.average(points, axis=0)
    pds = []
    for point in points:
        pds.append(calculate_distance(cm, point, helping_factor=0))
    d = calculate_distance(p, cm, helping_factor=np.average(pds))
    if d < current_best[2]:
        current_best = (category, idx, d)
    return current_best


def str_to_point(point_as_str):
    return np.asarray(list(map(int, point_as_str.split(','))))


def measure_distance_to_ball(category, current_best, idx, p, shape):
    r = shape['Radius']
    center = np.asarray(str_to_point(shape['Center']))
    d = calculate_distance(center, p, r)
    if d < current_best[2]:
        current_best = (category, idx, d)
    return current_best


def measure_distance_to_wall(category, current_best, idx, p, shape):
    points = []
    for key, point in shape.items():
        if key not in ('A', 'B'):
            continue
        points.append(str_to_point(point))
    d = norm(np.cross(points[1] - points[0], points[0] - p)) / norm(points[1] - points[0])
    if d < current_best[2]:
        current_best = (category, idx, d)
    return current_best


def calculate_distance(p1, p, helping_factor):
    diff = np.asarray(p) - p1
    d = np.sqrt(np.dot(diff.T, diff)) - helping_factor
    return d


def prepare_cart(bbox, output_dict, pred):
    tx1, tx2, tx3, tx4, ty1, ty2, ty3, ty4, txc1, tyc1, txc2, tyc2, tr = get_cart_params_test(bbox, get_pred_slice(pred,
                                                                                                                   "cart"))
    output_dict['Carts'].append({
        "A": xy_to_str(tx1, ty1),
        "B": xy_to_str(tx2, ty2),
        "C": xy_to_str(tx3, ty3),
        "D": xy_to_str(tx4, ty4),
        "wheel1": xy_to_str(txc1, tyc1),
        "wheel2": xy_to_str(txc2, tyc2),
        "radius": tr
    })


def prepare_ball(bbox, output_dict, pred, category_id):
    cat = 'static_ball' if category_id == 3 else 'ball'
    tr, tx, ty = get_ball_params_test(bbox, get_pred_slice(pred, cat))
    output_dict['Balls'].append({
        "Center": xy_to_str(tx, ty),
        "Radius": tr,
        "IsStatic": category_id == 3
    })


def prepare_wall(bbox, output_dict, pred, category_id):
    cat = 'ceiling' if category_id == 4 else 'floor'
    tx1, tx2, ty1, ty2 = get_line_params_test(bbox, get_pred_slice(pred, cat))
    output_dict['Walls'].append({
        "A": xy_to_str(tx1, ty1),
        "B": xy_to_str(tx2, ty2),
    })


def prepare_triangle(bbox, output_dict, pred):
    tx1, tx2, tx3, ty1, ty2, ty3 = get_triangle_params_test(bbox, get_pred_slice(pred, "triangle"))
    output_dict['Triangles'].append({
        "A": xy_to_str(tx1, ty1),
        "B": xy_to_str(tx2, ty2),
        "C": xy_to_str(tx3, ty3),
    })


def xy_to_str(x, y):
    return f"{x}, {y}"


def predict(im_ann, im_view, model):
    X = prepare_input(im_ann, im_view)
    predictions = model.forward(X) if model else X
    return predictions


def prepare_input(im_ann, im_view):
    X = []
    for ia in im_ann:
        ci = crop_img(im_view, ia)
        X.append(ci.flatten())
    X = torch.tensor(np.asarray(X), dtype=torch.float32, requires_grad=True)
    return X


def prepare_annotations(image_path):
    image_dir, image_name = os.path.split(image_path)
    im_view = np.asarray(ImageOps.grayscale(Image.open(image_path)))
    with open(os.path.join(image_dir, "our_json_new_result.bbox.json")) as ann:
        annotations = json.load(ann)

    # image_list, annotations, categories = get_dataset(image_dir)
    #
    # img_info = list(filter(lambda im: im['file_name'] == image_name, image_list))[0]
    # im_ann = list(filter(lambda im: im['image_id'] == img_info['id'], annotations))
    return annotations, im_view


def dump_output_to_json(output_dict, output_path):
    with open(os.path.join(output_path, 'example.json'), 'w') as out_json:
        json.dump(output_dict, out_json)


def make_output_obj():
    return dict(Carts=[], Balls=[], Walls=[], Blocks=[], Lines=[], Springs=[], Triangles=[])
