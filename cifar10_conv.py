import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
import torchvision.datasets
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from torchvision.transforms import ToTensor
from os.path import expanduser
import os
import tarfile
import torchvision.transforms as tt

loadSaveFile = False

dataFolder = expanduser("~/datasets")
saveFile = expanduser("~/datasets/cifar10-save.dat")

if not os.path.exists(os.path.join(dataFolder, "cifar10")):
    download_url("https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz", root=dataFolder)

    with tarfile.open(expanduser("~/datasets/cifar10.tgz"), "r:gz") as tar:
        tar.extractall(dataFolder)

trainingTransform = tt.Compose([
    tt.RandomCrop(size=32, padding=2, padding_mode="reflect"),
    tt.RandomHorizontalFlip(),
    tt.RandomGrayscale(p=0.05),
    tt.ToTensor()]
)

train_ds = ImageFolder(os.path.join(dataFolder, "cifar10", "train"), transform=trainingTransform)
test_ds = ImageFolder(os.path.join(dataFolder, "cifar10", "test"), transform=ToTensor())

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

class Cfar10Dataset(nn.Module):
    def __init__(self):
        super(Cfar10Dataset, self).__init__()
        self.initLayer = convolution(3, 32, True) # 16x16
        self.conv1 = nn.Sequential(convolution(32, 32), convolution(32, 32))
        self.middleLayer = convolution(32, 64, True) # 8x8
        self.conv2 = nn.Sequential(convolution(64, 64), convolution(64, 64))
        self.crossLayer = convolution(32, 64, True)
        self.classifier = nn.Sequential(
            nn.MaxPool2d(8),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        out = self.initLayer(x)
        crossOut = self.crossLayer(out)
        out = self.conv1(out) + out
        out = self.middleLayer(out)
        out = self.conv2(out) + out + crossOut
        return self.classifier(out)


def getDefaultDevice():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


device = getDefaultDevice()
print("device=", device)

def toDevice(data, device):
    if isinstance(data, list):
        return [toDevice(d, device) for d in data]
    else:
        return data.to(device)

class DeviceDataLoader:
    def __init__(self, device, dataLoader):
        self.device = device
        self.dataLoader = dataLoader

    def __len__(self):
        return len(self.dataLoader)

    def __iter__(self):
        for d in self.dataLoader:
            yield toDevice(d, self.device)


train_dl = DeviceDataLoader(device, train_dl)
test_dl = DeviceDataLoader(device, test_dl)

numEpoch = 20
model = Cfar10Dataset()
model.to(device)

if loadSaveFile:
    model.load_state_dict(torch.load(saveFile))

optimizer = torch.optim.SGD(model.parameters(), lr=10e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 10e-2, epochs=numEpoch, steps_per_epoch=len(train_dl))


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
        losses.append(loss)

        total += y.size(0)
        _, yMax = torch.max(yPredicted, dim=1)
        correct += (yMax == y).sum().item()
    avgLoss = torch.stack(losses).mean()
    accuracy = correct / total * 100
    print("loss=", avgLoss, "acc=", accuracy,"%")



torch.save(model.state_dict(), saveFile)