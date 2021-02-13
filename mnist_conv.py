import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
import torchvision.datasets
import torch.nn as nn
from torchvision.transforms import ToTensor
from os.path import expanduser
import os

loadSaveFile = False
saveFile = expanduser("~/datasets/mnist-save.dat")

dataFolder = expanduser("~/datasets")
os.makedirs(dataFolder, exist_ok=True)


train_ds = torchvision.datasets.MNIST(root=dataFolder, download=True, train=True, transform=ToTensor())
test_ds = torchvision.datasets.MNIST(root=dataFolder, download=True, train=False, transform=ToTensor())

train_dl = DataLoader(dataset=train_ds, batch_size=128, shuffle=True)
test_dl = DataLoader(dataset=train_ds, batch_size=128)

def convolution(inChannels, outChannels, useMaxPool=False):
    conv = nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1)
    bn = nn.BatchNorm2d(num_features=outChannels)
    relu = nn.ReLU()

    if useMaxPool:
        return nn.Sequential(conv, bn, relu, nn.MaxPool2d(2))
    else:
        return nn.Sequential(conv, bn, relu)

class MnistDataset(nn.Module):
    def __init__(self):
        super(MnistDataset, self).__init__()
        self.initLayer = convolution(1, 32, True) # 14x14
        self.conv1 = nn.Sequential(convolution(32, 32), convolution(32, 32))
        self.middleLayer = convolution(32, 64, True) # 7x7
        self.conv2 = nn.Sequential(convolution(64, 64), convolution(64, 64))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(7),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        out = self.initLayer(x)
        out = self.conv1(out) + out
        out = self.middleLayer(out)
        out = self.conv2(out) + out
        return self.classifier(out)


numEpoch = 10
model = MnistDataset()

if loadSaveFile:
    model.load_state_dict(torch.load(saveFile))

optimizer = torch.optim.SGD(model.parameters(), lr=10e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 10e-2, epochs=numEpoch, steps_per_epoch=len(train_dl))



def calculateAccuracy(yPredicted, y):
    _, yMax = torch.max(yPredicted, dim=1)
    return torch.tensor(torch.sum(yMax == y).item() / len(y))


for epoch in range(numEpoch):
    model.train()
    i = 0
    for x, y in train_dl:
        optimizer.zero_grad()
        yPredicted = model(x)
        loss = cross_entropy(yPredicted, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i%20 == 0:
            print("loss=", loss, "num=",i,"/",len(train_dl))
        i = i + 1

    model.eval()
    losses = []
    correct = 0
    total = 0
    for x, y in test_dl:
        yPredicted = model(x)
        loss = cross_entropy(yPredicted, y).detach()
        accuracy = calculateAccuracy(yPredicted, y)
        losses.append(loss)

        total += y.size(0)
        _, yMax = torch.max(yPredicted, dim=1)
        correct += (yMax == y).sum().item()
    avgLoss = torch.stack(losses).mean()
    avgAccuracy = correct / total * 100
    print("loss=", avgLoss, "acc=", avgAccuracy,"%")

torch.save(model.state_dict(), saveFile)