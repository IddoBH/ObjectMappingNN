import os

from PIL import Image, ImageDraw

DATASET_PATH = ""
OUT_PATH = "/Users/iddobar-haim/PycharmProjects/ObjectMappingNN/outputs"
OUT_NAME = "cwr"

def test_loop(model, X, annotations, image_list):
    predictions = model.forward(X)
    image_dict = {}
    for pred, ann in zip(predictions, annotations):
        if ann['category_id'] == 3:
            center = pred[0:2]
        elif ann['category_id'] == 6:
            center = pred[2:4]
        bbox = ann['bbox']
        tx = round(center[0].item() * (bbox[2] / 32) + bbox[0])
        ty = round(center[1].item() * (bbox[3] / 32) + bbox[1])
        if ann['image_id'] not in image_dict:
            image_dict[ann['image_id']] = []
        image_dict[ann['image_id']].append((tx, ty))

    out_dir = os.path.join(OUT_PATH, OUT_NAME)
    os.mkdir(out_dir)

    for im_id, centers in image_dict.items():
        # Create Image object
        im_info = list(filter(lambda im: im['id'] == im_id, image_list))[0]
        im = Image.open(os.path.join(DATASET_PATH, 'val', im_info['file_name']))

        # Draw line
        draw = ImageDraw.Draw(im)
        draw.point(centers, fill='red')

        # Show image
        im.save(os.path.join(out_dir, im_info['file_name']))
