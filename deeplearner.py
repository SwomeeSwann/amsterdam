import torch
import torch.nn as nn
import cv2 as cv
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt

images1 = ["faces/s1/1.pgm", "faces/s6/3.pgm", "faces/s30/7.pgm", "faces/s19/9.pgm", "faces/s2/4.pgm", "faces/s7/3.pgm"]
images2 = ["faces/s1/9.pgm", "faces/s2/4.pgm", "faces/s4/3.pgm", "faces/s19/7.pgm", "faces/s3/4.pgm", "faces/s7/10.pgm"]
labels = [1, 0, 0, 1, 0, 1]

class SiameseDataset():
    def __init__(self) -> None:
        pass

    def getDataSet():
        randomSet = random.randint(0, len(labels) - 1)
        img1 = cv.imread(images1[randomSet], cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(images2[randomSet], cv.IMREAD_GRAYSCALE)

        return img1, img2, labels[randomSet]


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
data = SiameseDataset
contrastive_loss = ContrastiveLoss()

optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

num_epochs = int(input("How many epochs would you like to train?"))

negative_values = []
positive_values = []

for epoch in range(num_epochs):
    
    img1, img2, label = data.getDataSet()

    optimizer.zero_grad()

    img_tensor = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0).float()
    img_tensor2 = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).float()


    x = network.forward_once(img_tensor)
    y = network.forward_once(img_tensor2)

    label = torch.tensor([label])
    loss = contrastive_loss(x, y, label)

    loss.backward()
    optimizer.step()
    print(label.item(), loss.item())
    if label.item() == 0:
        negative_values.append(loss.item())
    else:
        positive_values.append(loss.item())
        
    

plt.plot(negative_values, marker='o', linestyle='-')
plt.grid(True)
plt.title("Negative")
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.plot(positive_values, marker='x', linestyle='-')
plt.grid(True)
plt.title("Negative")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
