import os

import torch

from dataset_maker import get_dataset, make_tensors, DATASET_PATH_train, DATASET_PATH_train_gen, MASKS
from mapper import objectMapper

MODEL_SAVE_DIR_PATH = "/home/shovalg@staff.technion.ac.il/PycharmProjects/ObjectMappingNN/models"


def train_loop(model, X, y, mask, criterion, optimizer, epochs):
    for ep in range(epochs):
        preds = model.forward(X)
        preds = preds * mask
        loss = criterion(preds, y)
        print(f"Epoch: {ep + 1}", f"Loss: {loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train(dataset, device):
    print("getting data")
    train_image_list, train_annotations, train_categories = get_dataset(dataset)
    print("making tensors")
    train_X, train_y, train_mask = make_tensors(train_image_list, train_annotations, dataset)
    print("training")
    train_loop(model, train_X.to(device), train_y.to(device), train_mask.to(device), criterion, optimizer, epochs)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print("making nn")
    model = objectMapper(len(MASKS["ball"]))
    model.to(device)
    print("prep")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    epochs = 3000
    print("training real")
    train(DATASET_PATH_train, device)
    for i in range(1, 101):
        print(f'running gen set_{i}')
        train(os.path.join(DATASET_PATH_train_gen, 'train', f'set_{i}'), device)
    print("training real")
    train(DATASET_PATH_train, device)
    print("Saving")
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR_PATH, "total_1.pth"))
    print("Done!")

