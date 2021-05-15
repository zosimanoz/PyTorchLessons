
#################
# Saving and Loading Model
#################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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
model = CNN()
x = torch.randn(64, 1, 28, 28)
print(model(x).shape)
'''

# Utility Functions
def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("=> saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparams
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
epochs = 5
load_model = True

# Load aata
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = CNN().to(device=device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth"))

# Train network
for epoch in range(epochs):
    losses = []
    accuracies = []

    if epoch % 3 == 0:
        checkpoint = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
        save_checkpoint(checkpoint)

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device=device)
        target = target.to(device=device)

        # forward propagation
        scores = model(data)
        loss = criterion(scores, target)
        # append all losses
        losses.append(loss.item())

        # backward propagation
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()

        # calculate running training accuracry
        _, predictions = scores.max(1)
        num_correct = (predictions == target).sum()

        running_training_acc = float(num_correct) / float(data.shape[0])
        accuracies.append(running_training_acc)

    mean_loss = sum(losses) / len(losses)
    accuracy = sum(accuracies) / len(accuracies)
    print(f'Loss at epoch {epoch} is {mean_loss:.5f} and accuracry : {accuracy}')

