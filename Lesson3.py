# Convolutional Neural Network
# Same convolultion Formula:

'''
    n_out = (n_in + 2p - K)/s + 1
    n_in : no of input features
    n_out: no of output features
    k: convolution kernel size
    p: convolution padding size
    s: convolution stride size
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# create a CNN
class CNN(nn.Module):
    def __init__(self, in_channles = 1, num_classes = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparams
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
epochs = 10

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
loss = 0

# Train network
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device=device)
        target = target.to(device=device)

        # forward propagation
        scores = model(data)
        loss = criterion(scores, target)

        # backward propagation
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()

    print(f"Epoch {epoch + 1} / {epochs}, Loss: {loss}")


# Check acc on training and test data to see how good is our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('checking acc on train data')
    else:
        print('checking acc on test data')

    num_correct = 0
    num_samples = 0
    model.eval()

    # while testing we are not computing gradients
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            # shape of scors = 64x10, so we want max of second dimensions for accuracry
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracry {float(num_correct) / float(num_samples) * 100:.2f}')
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
