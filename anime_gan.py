import torch
from torch.nn.functional import cross_entropy, binary_cross_entropy
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
from torchvision.utils import save_image

batchSize=128

loadSaveFile=True
discriminatorSaveFile=expanduser("~/datasets/save-discriminator-30")
generatorSaveFile=expanduser("~/datasets/save-generator-30")

datasetsFolder = expanduser("~/datasets")
dataFolder = expanduser("~/datasets/animefaces")
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

trainTransform = tt.Compose([
    tt.Resize(64),
    tt.CenterCrop(64),
    tt.RandomHorizontalFlip(),
    tt.ToTensor(),
    tt.Normalize(*stats)
])

train_ds = ImageFolder(dataFolder, transform=trainTransform)
train_dl = DataLoader(train_ds, batch_size=batchSize, pin_memory=True, shuffle=True)


class GanDiscriminatorModel(nn.Module):
    def __init__(self):
        super(GanDiscriminatorModel, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 64, stride=2, padding=1, kernel_size=3), # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, stride=2, padding=1, kernel_size=3),  # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, stride=2, padding=1, kernel_size=3),  # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, stride=2, padding=1, kernel_size=3),  # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1, stride=2, padding=0, kernel_size=4), # 512x1x1
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class GanGeneratorModel(nn.Module):
    def __init__(self, generatorSize):
        super(GanGeneratorModel, self).__init__()

        self.network = nn.Sequential(
            nn.ConvTranspose2d(generatorSize, 512, kernel_size=4, stride=1, padding=0, bias=False), # generatorSize x 1 x 1 -> 512 x 4 x 4
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),  # 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 256 x 16 x 16
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),  # 3 x 64 x 64

            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)


numEpoch=20
hiddenSize=100

generatorModel = GanGeneratorModel(hiddenSize)
discriminatorModel = GanDiscriminatorModel()

generatorOptimizer = torch.optim.Adam(generatorModel.parameters(), lr = 0.0002, betas=(0.5, 0.999))
discriminatorOptimizer = torch.optim.Adam(discriminatorModel.parameters(), lr = 0.0002, betas=(0.5, 0.999))

if loadSaveFile:
    generatorModel.load_state_dict(torch.load(generatorSaveFile, map_location=device))
    discriminatorModel.load_state_dict(torch.load(discriminatorSaveFile, map_location=device))


def trainDiscriminator(x, optimizer):
    realImageResult = discriminatorModel(x)
    realImageExpected = torch.ones(len(x), 1)
    realLoss = binary_cross_entropy(realImageResult, realImageExpected)

    generatorInput = torch.randn(len(x), hiddenSize, 1, 1)
    fakeImages = generatorModel(generatorInput)
    fakeImageResult = discriminatorModel(fakeImages)
    fakeImageExpected = torch.zeros(len(x), 1)
    fakeLoss = binary_cross_entropy(fakeImageResult, fakeImageExpected)

    loss = realLoss + fakeLoss
    loss.backward()
    optimizer.step()

def trainGenerator(optimizer):
    generatorInput = torch.randn(batchSize, hiddenSize, 1, 1)
    fakeImages = generatorModel(generatorInput)
    fakeExpected = torch.ones(batchSize, 1)
    output = discriminatorModel(fakeImages)
    loss = binary_cross_entropy(output, fakeExpected)
    loss.backward()
    optimizer.step()

def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]

def save_samples(index, progress):
    fake_images = generatorModel(torch.randn(batchSize, hiddenSize, 1, 1))
    fake_fname = 'generated-images-{0:0=4d}-{1:0=4d}.png'.format(index, progress)
    save_image(denorm(fake_images[:32]), os.path.join(expanduser("~/datasets/animeout"), fake_fname), nrow=8)
    print('Saving', fake_fname)

save_samples(0, 0)

for epoch in range(numEpoch):
    generatorModel.train()
    discriminatorModel.train()
    i = 0
    for x, y in train_dl:
        generatorOptimizer.zero_grad()
        discriminatorOptimizer.zero_grad()

        trainDiscriminator(x, discriminatorOptimizer)
        trainGenerator(generatorOptimizer)
        i = i + 1
        if i % 20 == 0:
            print("epoch=", epoch," ", i, " / ", len(train_dl))
            save_samples(epoch + 1, i)

    save_samples(epoch + 1, 9999)
    if epoch % 10 == 0:
        torch.save(generatorModel.state_dict(), os.path.join(datasetsFolder, "save-generator-" + str(epoch)))
        torch.save(discriminatorModel.state_dict(), os.path.join(datasetsFolder, "save-discriminator-" + str(epoch)))

torch.save(generatorModel.state_dict(), os.path.join(datasetsFolder, "save-generator-" + str(numEpoch)))
torch.save(discriminatorModel.state_dict(), os.path.join(datasetsFolder, "save-discriminator-" + str(numEpoch)))