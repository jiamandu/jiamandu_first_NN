import torch
import torchvision
from torch import nn
from torch.nn import Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

data_set_train = torchvision.datasets.CIFAR10(root='Dataset',
                                              train=True,
                                              download=True,
                                              transform=torchvision.transforms.ToTensor())

data_set_test = torchvision.datasets.CIFAR10(root='Dataset',
                                              train=False,
                                              download=True,
                                              transform=torchvision.transforms.ToTensor())

loader_train = DataLoader(data_set_train,shuffle=True,batch_size=1,drop_last=True)

loader_test = DataLoader(data_set_test,shuffle=True,batch_size=1,drop_last=True)

class Mynn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1      =   nn.Conv2d(in_channels=3,
                                      out_channels=32,
                                      kernel_size=5,
                                      padding=2)

        self.maxpool1   =   nn.MaxPool2d(kernel_size=2)

        self.conv2      =   nn.Conv2d(in_channels=32,
                                      out_channels=32,
                                      kernel_size=5,
                                      padding=2)

        self.maxpool2   =   nn.MaxPool2d(kernel_size=2)

        self.conv3      =   nn.Conv2d(in_channels=32,
                                      out_channels=64,
                                      kernel_size=5,
                                      padding=2)

        self.maxpool3   =   nn.MaxPool2d(kernel_size=2)

        self.flaten1    =   nn.Flatten()

        self.liner1     =   nn.Linear(in_features=1024,
                                      out_features=64)

        self.liner2     =   nn.Linear(in_features=64,
                                      out_features=10)

        self.module1 = Sequential(
                                nn.Conv2d(in_channels=3,
                                          out_channels=32,
                                          kernel_size=5,
                                          padding=2),

                                nn.MaxPool2d(kernel_size=2),

                                nn.Conv2d(in_channels=32,
                                          out_channels=32,
                                          kernel_size=5,
                                          padding=2),

                                nn.MaxPool2d(kernel_size=2),

                                nn.Conv2d(in_channels=32,
                                          out_channels=64,
                                          kernel_size=5,
                                          padding=2),

                                nn.MaxPool2d(kernel_size=2),

                                nn.Flatten(),

                                nn.Linear(in_features=1024,
                                          out_features = 64),

                                nn.Linear(in_features=64,
                                          out_features=10),
        )

    def forward(self,x):
        x =self.module1(x)
        return x

mynn = Mynn()

loss = nn.CrossEntropyLoss()

optim = torch.optim.SGD(mynn.parameters(),lr=0.01)


for epoch in range(20):

    running_loss = 0

    for data in loader_test:
        images,targets = data
        output = mynn(images)

        loss_cross = loss(output,targets)
        optim.zero_grad()
        loss_cross.backward()
        optim.step()

        running_loss = running_loss + loss_cross

    print(running_loss)