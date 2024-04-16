import torch
import torch.nn as nn
import cv2 as cv
import random
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=10),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=4),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(6144, 1024),  # Reduce the number of neurons
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),  # Reduce the number of neurons further
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),  # Further reduction
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),  # Final layer with even fewer neurons
            nn.Sigmoid())

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        print(output.size())
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        return output1

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0) -> None:
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        euclidean_dist = F.pairwise_distance(out1, out2)
        loss_contrast = torch.mean((1 - label) * torch.pow(euclidean_dist, 2)
            + (label) * torch.pow(torch.clamp(self.margin - euclidean_dist, min=0.0), 2))
        
        return loss_contrast


network = SiameseNetwork()
contrastive_loss = ContrastiveLoss()

optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

img = cv.imread("faces/s1/1.pgm", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("faces/s1/2.pgm", cv.IMREAD_GRAYSCALE)

img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
img_tensor2 = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).float()


x = network.forward_once(img_tensor)
y = network.forward_once(img_tensor2)

label = torch.tensor([1])
loss = contrastive_loss(x, y, label)

loss.backward()
optimizer.step()

print(loss)

    