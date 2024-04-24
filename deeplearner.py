import torch
import torch.nn as nn
import cv2 as cv
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Remove unnecessary import
# import faceDetect as fd # type: ignore

# Define image paths and labels
images1 = ["faces/s1/1.pgm", "faces/s6/3.pgm", "faces/s30/7.pgm", "faces/s19/9.pgm", "faces/s2/4.pgm", "faces/s7/3.pgm", "faces/s37/2.pgm", "faces/s26/8.pgm", "faces/s40/1.pgm", "faces/s28/5.pgm", "faces/s40/6.pgm", "faces/s8/5.pgm", "faces/s17/2.pgm", "faces/s19/3.pgm"]
images2 = ["faces/s1/9.pgm", "faces/s2/4.pgm", "faces/s4/3.pgm", "faces/s19/7.pgm", "faces/s3/4.pgm", "faces/s7/10.pgm", "faces/s5/9.pgm", "faces/s26/3.pgm", "faces/s40/10.pgm", "faces/s21/1.pgm", "faces/s3/10.pgm", "faces/s8/6.pgm", "faces/s20/9.pgm", "faces/s19/10.pgm"]
labels = [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]

class SiameseDataset():
    def __init__(self, images1, images2, labels) -> None:
        self.images1 = images1
        self.images2 = images2
        self.labels = labels

    def getDataSet(self):
        randomSet = random.randint(0, len(self.labels) - 1)
        img1 = cv.imread(self.images1[randomSet], cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(self.images2[randomSet], cv.IMREAD_GRAYSCALE)

        img1 = cv.resize(img1, (100, 100))
        img2 = cv.resize(img2, (100, 100))

        return img1, img2, self.labels[randomSet]

    def getTest(self):
        img1 = cv.imread("faces/s40/1.pgm", cv.IMREAD_GRAYSCALE)
        img2 = cv.imread("faces/s40/3.pgm", cv.IMREAD_GRAYSCALE)

        img1 = cv.resize(img1, (100, 100))
        img2 = cv.resize(img2, (100, 100))

        return img1, img2

# Instantiate SiameseDataset
data = SiameseDataset(images1, images2, labels)

# Define SiameseNetwork architecture
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 8, kernel_size=3),  # Increased number of filters
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 16, kernel_size=3),  # Increased number of filters
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 16, kernel_size=3),  # Increased number of filters
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
        )

        # Add regularization
        # Example: Dropout
        self.fc1 = nn.Sequential(
            nn.Linear(16*100*100, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Add dropout layer

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Add dropout layer

            nn.Linear(500, 5)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# Define ContrastiveLoss class
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive


# Instantiate SiameseNetwork and load saved state
network = SiameseNetwork()
#network.load_state_dict(torch.load("state.pt"))

# Define optimizer and loss function
optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
contrastive_loss = ContrastiveLoss()

# Lists to store loss values
loss_val = []

# Training loop
def train(num_epochs):
    for epoch in range(num_epochs):
        for i in range(int(len(labels) / 2)):
            img1, img2, label = data.getDataSet()

            optimizer.zero_grad()

            img_tensor = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimension
            img_tensor2 = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimension

            out1, out2 = network.forward(img_tensor, img_tensor2)

            label = torch.tensor([label], dtype=torch.float32)  # Convert label to float tensor


            loss = contrastive_loss(out1, out2, label)

            loss.backward()
            optimizer.step()
            print(label.item(), loss.item())
            loss_val.append(loss.item())


# Plot loss values
def graph(values):
    plt.plot(values, marker='o', linestyle='-', label='Negative Loss')
    plt.grid(True)
    plt.title("Loss Value Over Training")
    plt.xlabel('Set')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    num_epochs = int(input("How many epochs would you like to train?"))
    train(num_epochs)
    graph(loss_val)
    torch.save(network.state_dict(), "state.pt")

    network.load_state_dict(torch.load("state.pt"))

    network.eval()

    img, img2 = data.getTest()

    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()  
    img_tensor2 = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).float()  

    out, out2 = network(img_tensor, img_tensor2)

    loss = F.pairwise_distance(out, out2)

    print(loss.item())