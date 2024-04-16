import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2

# Siamese network architecture
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10),
            nn.Linear(10, 2)
        )

    def forward_once(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# Custom dataset for loading pairs of images
class SiameseDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform

    def __getitem__(self, index):
        img1_path = self.image_folder[index][0]
        img2_path = self.image_folder[index][1]
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # Assuming the last element in each pair in train_image_folder is the label
        label = self.image_folder[index][1]

        return img1, img2, label

    def __len__(self):
        return len(self.image_folder)

# Custom contrastive loss function for Siamese network
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - target) * torch.pow(euclidean_distance, 2) +
                                      (target) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# Training the Siamese network
def train_siamese(train_loader, model, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            img1, img2, labels = data
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

# Example usage
if __name__ == "__main__":
    # Assuming you have a folder containing pairs of images for training
    train_image_folder = [
        ("faces/s1/1.pgm", "faces/s1/4.pgm", ),
        ("faces/s4/1.pgm", "faces/s7/1.pgm"),
        ("faces/s4/1.pgm", "faces/s6/1.pgm"),
        # Add more pairs as needed
    ]

    # Define transformations for images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((100, 100))  # Resize images to a consistent size
    ])

    # Create dataset and dataloader
    train_dataset = SiameseDataset(train_image_folder, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize Siamese network, loss function, and optimizer
    siamese_net = SiameseNetwork()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(siamese_net.parameters(), lr=0.001)

    # Train the Siamese network
    train_siamese(train_loader, siamese_net, criterion, optimizer, epochs=10)
