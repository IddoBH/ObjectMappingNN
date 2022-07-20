from torch import nn
import torch.nn.functional as F

from dataset_maker import ROI_SIZE


class objectMapper(nn.Module):
    def __init__(self, output_size):
        super(objectMapper, self).__init__()
        self.fc1 = nn.Linear(ROI_SIZE * ROI_SIZE, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



