import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.mode == "test":
            x = F.softmax(x, dim=-1)
        return x


if __name__ == "__main__":
    lenet = LeNet5("test")
    lenet.load_state_dict(torch.load("lenet.pth", map_location="cpu"))
    lenet.eval()
    torch.onnx.export(
        lenet,
        torch.randn(1, 1, 28, 28),
        "lenet.onnx",
        verbose=True,
        opset_version=11,
    )
