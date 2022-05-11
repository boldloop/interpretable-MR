import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.progress import track
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

torch.manual_seed(42)

# configure argparse for Farnam
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--save_model", type=bool, default=True)
args = parser.parse_args()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
img_scale = 2


# import data and create trainloader
class Cropper:
    def __call__(self, img):
        return transforms.functional.crop(img, 35, 54, 218, 336)


# import data and create trainloader
train_dataset = datasets.ImageFolder(
    "data/train",
    transform=transforms.Compose(
        [
            transforms.Grayscale(),
            Cropper(),
            transforms.Resize((218 // img_scale, 336 // img_scale)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5)
        ]
    ),
)
test_dataset = datasets.ImageFolder(
    "data/test",
    transform=transforms.Compose(
        [
            transforms.Grayscale(),
            Cropper(),
            transforms.Resize((218 // img_scale, 336 // img_scale)),
            transforms.ToTensor(),
        ]
    ),
)

train_data = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
train_labels = torch.Tensor([train_dataset[i][1] for i in range(len(train_dataset))])

# plot transformed spectrograms
# plt.imshow(np.squeeze(train_data[0]))
# plt.show()
# exit()

test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
test_labels = torch.Tensor([test_dataset[i][1] for i in range(len(test_dataset))])


trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)

# construct CNN
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool = nn.MaxPool2d(3, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 20, 5)
        self.conv3 = nn.Conv2d(20, 20, 3)
        self.conv4 = nn.Conv2d(20, 20, 3)
        self.fc1 = nn.Linear(3960, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn(self.conv1(x))))
        res1 = x.view(x.shape[0], -1).clone()
        x = self.pool(F.relu(self.bn(self.conv2(x))))
        res2 = x.view(x.shape[0], -1).clone()
        x = self.pool2(F.relu(self.bn(self.conv3(x))))
        res3 = x.view(x.shape[0], -1).clone()
        x = F.relu(self.bn(self.conv4(x)))
        x = x.view(x.shape[0], -1)
        x = torch.cat((x, res2, res3), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=learning_rate
)
model.load_state_dict(torch.load('torch_model.pth'))

for epoch in track(range(num_epochs)):
#     for batch_index, (images, labels) in enumerate(trainloader):

#         images = images.to(device)
#         labels = labels.to(device)

#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()



    if (epoch + 1) % 1 == 0:
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(10)]
            n_class_samples = [0 for i in range(10)]
            pred = []
            true_labels = []
            for images, labels in testloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                # max returns (value ,index)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()
                pred.extend(predicted.detach().numpy())
                true_labels.extend(labels.detach().numpy())
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
            # Display confusion matrix
            categories = np.array(["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"])
            classes = [str(x) for x in range(10)]
            cm = confusion_matrix(pred, true_labels)
            
            p = sns.heatmap(cm, cmap='coolwarm', annot=True, fmt='g')
            p.set_xticklabels(categories, rotation=30)
            p.set_yticklabels(categories, rotation=30)
            p.set_title(f"Confusion matrix (test data)", fontsize=15)
            plt.savefig("./conf_mat.jpeg")

if args.save_model:
    torch.save(model.state_dict(), "./torch_model.pth")
