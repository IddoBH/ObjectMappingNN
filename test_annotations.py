from dataset_maker import get_dataset, make_tensors
from test import test_loop

dsp = "/Users/iddobar-haim/Library/CloudStorage/GoogleDrive-idodibh@gmail.com/My Drive/ARP/rect/train"

OUTOPUT_DIR = "/Users/iddobar-haim/PycharmProjects/ObjectMappingNN/outputs/ta"

if __name__ == '__main__':
    image_list, annotations, categories = get_dataset(dsp)
    X, y, mask = make_tensors(image_list, annotations)
    test_loop(None, y, annotations, image_list, dataset_path=dsp, out_dir=OUTOPUT_DIR)
