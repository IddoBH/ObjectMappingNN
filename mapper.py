from torch import nn
import torch.nn.functional as F

from dataset_maker import ROI_SIZE


class objectMapper(nn.Module):
    def __init__(self, output_size):
        super(objectMapper, self).__init__()
        self.fc1 = nn.Linear(ROI_SIZE * ROI_SIZE, 2**12)
        self.fc2 = nn.Linear(2**12, 2**11)
        self.fc3 = nn.Linear(2 ** 11, 2 ** 10)
        self.fc4 = nn.Linear(2 ** 10, 2 ** 8)
        self.fc5 = nn.Linear(2 ** 8, 2 ** 6)
        self.fc6 = nn.Linear(2 ** 6, 2 ** 6)
        self.fc7 = nn.Linear(2 ** 6, 2 ** 5)
        self.fc8 = nn.Linear(2 ** 5, 2 ** 5)
        self.fc_out = nn.Linear(2 ** 5, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc_out(x)
        return x



