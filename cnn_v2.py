import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.progress import track
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import matplotlib.pyplot as plt
import numpy as np

# configure argparse for Farnam
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate

# import data and create trainloader
class Cropper:
    def __call__(self, img):
        return transforms.functional.crop(img, 35, 54, 218, 336)

# import data and create trainloader
train_dataset = datasets.ImageFolder(
    "data/train",
    transform=transforms.Compose(
        [transforms.Grayscale(), Cropper(),  transforms.ToTensor()]
    ),
)
test_dataset = datasets.ImageFolder(
    "data/test",
    transform=transforms.Compose(
        [transforms.Grayscale(), Cropper(),  transforms.ToTensor()]
    ),
)

train_data = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
train_labels = torch.Tensor([train_dataset[i][1] for i in range(len(train_dataset))])

plt.imshow(np.squeeze(train_data[4]))
plt.show()
exit()

test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
test_labels = torch.Tensor([test_dataset[i][1] for i in range(len(test_dataset))])

trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)

# define CNN
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, 3, padding=0)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(5, 5)
        self.bn = nn.BatchNorm2d(30)
        self.conv2 = nn.Conv2d(30, 30, 3, padding=0)
        self.conv3 = nn.Conv2d(30, 30, 3, padding=0)
        self.fc1 = nn.Linear(240, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        # Implement skip connections by flattening and concatenating intermediate results
        x = self.pool1(F.relu(self.bn(self.conv1(x))))
        x = self.pool2(F.relu(self.bn(self.conv2(x))))
        x = self.pool1(F.relu(self.bn(self.conv3(x))))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=learning_rate, weight_decay=0
)

for epoch in track(range(num_epochs)):
    for batch_index, (specs, labels) in enumerate(trainloader):

        specs = specs.to(device)
        labels = labels.to(device)

        outputs = model(specs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(10)]
            n_class_samples = [0 for i in range(10)]
            for images, labels in testloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                # max returns (value ,index)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            test_acc = 100.0 * n_correct / n_samples
            for images, labels in trainloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                # max returns (value ,index)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            train_acc = 100.0 * n_correct / n_samples
            print(
                f"Epoch: {epoch+1}. Test acc: {test_acc} %. Train acc: {train_acc:.1f} %."
            )
