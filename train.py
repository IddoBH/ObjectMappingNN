import os

import torch

from dataset_maker import get_dataset, make_tensors, DATASET_PATH, MASKS
from mapper import objectMapper

MODEL_SAVE_DIR_PATH = "/Users/iddobar-haim/PycharmProjects/ObjectMappingNN/models"


def train_loop(model, X, y, mask, criterion, optimizer, epochs):
    for ep in range(epochs):
        preds = model.forward(X)
        preds = preds * mask
        loss = criterion(preds, y)
        print(f"Epoch: {ep + 1}", f"Loss: {loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    print("making nn")
    model = objectMapper(len(MASKS["ball"]))
    print("getting data")
    train_image_list, train_annotations, train_categories = get_dataset(os.path.join(DATASET_PATH, 'train'), prt=True)
    print("making tensors")
    train_X, train_y, train_mask = make_tensors(train_image_list, train_annotations, )
    print("prep")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=.9)
    epochs = 800
    print("training")
    train_loop(model, train_X, train_y, train_mask, criterion, optimizer, epochs)
    print("Saving")
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR_PATH, "circles_and_triangles.pth"))
    print("Done!")
