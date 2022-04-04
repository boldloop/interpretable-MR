import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # activation functions
import torch.optim as optim  # optimizer
from torch.autograd import Variable # add gradients to tensors
from torch.nn import Parameter # model parameter functionality

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ##################################
# import data
# ##################################
# load into torch datasets
train_dataset = datasets.ImageFolder('data/train',
    transform=transforms.Compose([transforms.Grayscale(),
                                  transforms.Resize((36, 54)),
                                  transforms.ToTensor()]))
test_dataset = datasets.ImageFolder('data/test',
    transform=transforms.Compose([transforms.Grayscale(),
                                  transforms.Resize((36, 54)),
                                  transforms.ToTensor()]))

train_data = torch.stack([train_dataset[i][0]
                          for i in range(len(train_dataset))])
train_labels = torch.Tensor([train_dataset[i][1]
                             for i in range(len(train_dataset))])

test_data = torch.stack([test_dataset[i][0]
                         for i in range(len(test_dataset))])
test_labels = torch.Tensor([test_dataset[i][1]
                            for i in range(len(test_dataset))])

# ##################################
# helper functions
# ##################################

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

def get_accuracy(output, targets):
    """calculates accuracy from model output and targets
    """
    output = output.detach()
    predicted = output.argmax(-1)
    correct = (predicted == targets).sum().item()

    accuracy = correct / output.size(0) * 100

    return accuracy

CELoss = torch.nn.CrossEntropyLoss()

# ##################################
# create a dataloader
# ##################################
trainloader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=32,
                                          shuffle=True)

# ##################################
# main training function
# ##################################

def train(model, num_epochs=100, lr=0.01):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # initialize loss list
    metrics = []

    # iterate over epochs
    for ep in range(num_epochs):
        model.train()

        # iterate over batches
        for batch_indx, batch in enumerate(trainloader):

            # unpack batch
            data, labels = batch

            optimizer.zero_grad()
            
            pred = model(data)
            loss = CELoss(pred, labels)

            loss.backward()
            optimizer.step()

        model.eval() # model will not calculate gradients for this pass, and will disable dropout
        train_ep_pred = model(train_data)
        test_ep_pred = model(test_data)

        train_accuracy = get_accuracy(train_ep_pred, train_labels)
        test_accuracy = get_accuracy(test_ep_pred, test_labels)

        # print accuracy every 100 epochs
        if ep % 1 == 0:
            print("train acc: {}\t test acc: {}\t at epoch: {}".format(train_accuracy,test_accuracy,ep))
        metrics.append([train_accuracy, test_accuracy])

    return np.array(metrics), model

model = nn.Sequential(nn.Conv2d(1, 6, 5),
                      nn.ReLU(),
                      nn.MaxPool2d(2, 2),
                      nn.Conv2d(6, 16, 5),
                      nn.ReLU(),
                      nn.MaxPool2d(2, 2),
                      Flatten(),
                      nn.Linear(960, 120),
                      nn.ReLU(),
                      nn.Linear(120, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10))

# done.
