import os

import torch

from dataset_maker import get_dataset, make_tensors, DATASET_PATH_train, DATASET_PATH_train_gen, MASKS, make_tv
from mapper import objectMapper
from matplotlib import pyplot as plt

MODEL_NAME = "rectangle_special_sort.pth"

MODEL_SAVE_DIR_PATH = "/home/shovalg@staff.technion.ac.il/PycharmProjects/ObjectMappingNN/models"


def train_loop(model, X_train, y_train, mask_train, X_val, y_val, mask_val, criterion, optimizer, epochs):
    train_losses = []
    val_losses = []
    for ep in range(epochs):
        loss_t = predict(X_train, y_train, mask_train, model, criterion)
        loss_v = predict(X_val, y_val, mask_val, model, criterion)
        train_losses.append(loss_t.item())
        val_losses.append(loss_v.item())
        print(f"Epoch: {ep + 1}", f"Train Loss: {loss_t}", f"Val Loss: {loss_v}")
        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.savefig(os.path.join(MODEL_SAVE_DIR_PATH, os.path.splitext(MODEL_NAME)[0] + "_train_res.png"))
    with open('/home/shovalg@staff.technion.ac.il/PycharmProjects/ObjectMappingNN/results.csv','a') as res_f:
        res_f.write(f'{MODEL_NAME},{val_losses[-1]}\n')





def predict(X, y, mask, model, criterion):
    preds_t = model.forward(X)
    preds_t = preds_t * mask
    return criterion(preds_t, y)


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
    epochs = 300000
    X_train, y_train, mask_train, X_val, y_val, mask_val = make_tv()
    train_loop(model, X_train.to(device), y_train.to(device), mask_train.to(device), X_val.to(device), y_val.to(device), mask_val.to(device), criterion, optimizer, epochs)
    print("Saving")
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR_PATH, MODEL_NAME))
    print("Done!")

