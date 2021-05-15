
#################
#  Custom Data Loaders
#################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from DataSetLoader import CatsAndDogsDataset

class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

'''
x = torch.randn(64, 1, 28, 28)
print(x.shape)
model = CNN()
print(model(x).shape)
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparams
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 8
epochs = 5
load_model = False


dataset = CatsAndDogsDataset(csv_file='cats_dog.csv',
                             root_dir='CatDogDataset',
                             transform=transforms.Compose([
                                 transforms.ToPILImage(),
                                 transforms.Resize(size=(100,100)),
                                 transforms.ToTensor()
                             ]))

train_set, test_set = torch.utils.data.random_split(dataset, [42, 10])
print(len(train_set), len(test_set))

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

for batch_index, (data, targets) in enumerate(train_loader):
    print(batch_index)

