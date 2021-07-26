import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import random
import os
import customDataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)

        self.dropout=nn.Dropout(0.25)

        self.fc1 = nn.Linear(96*96, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3= nn.Linear(100,10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*12*12)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x




def seed(seed_value):

    # This removes randomness, makes everything deterministic

    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(model, use_cuda, train_loader, optimizer, epoch):

    model.train()   # Tell the model to prepare for training
    print('training')

    for batch_idx, (data, target) in enumerate(train_loader):  # Get the batch


        if use_cuda:
            data, target = data.cuda(), target.cuda()  # Sending the data to the GPU
            # data, y_onehot = data.cuda(), y_onehot.cuda()  # Sending the data to the GPU

        optimizer.zero_grad()  # Setting the cumulative gradients to 0
        output = model(data)  # Forward pass through the model
        # loss = torch.mean((output - y_onehot) ** 2)  # Calculating the loss
        loss = F.cross_entropy(output, target)
        loss.backward()  # Calculating the gradients of the model. Note that the model has not yet been updated.
        optimizer.step()  # Updating the model parameters. Note that this does not remove the stored gradients!

        # batch_idx = train_loader.batch_size
        if (batch_idx) % 100 == 0 or batch_idx == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, use_cuda, test_loader):

    model.eval()  # Tell the model to prepare for testing or evaluation

    test_loss=0
    correct=0
    with torch.no_grad():  # Tell the model that gradients need not be calculated
        for data, target in test_loader:  # Get the batch

            
            if use_cuda:
                data, target = data.cuda(), target.cuda()  # Sending the data to the GPU

            # argmax([0.1, 0.2, 0.9, 0.4]) => 2
            # output - shape = [1000, 10], argmax(dim=1) => [1000]
            output = model(data)  # Forward pass
            test_loss += F.cross_entropy(output, target, reduction='sum')  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the maximum output
            correct += pred.eq(target.view_as(pred)).sum().item()  # Get total number of correct samples
    test_loss /= len(test_loader.dataset)  # Accuracy = Total Correct / Total Samples

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    


def main():

    use_cuda = False            # Set it to False if you are using a CPU

    seed(0)                 # Used to fix randomness in the code! Very important!

    train_dataset = customDataset.customDatasetClass('../train_data')
    train_loader = DataLoader(dataset=train_dataset,batch_size=10,shuffle=True,num_workers=2)
    test_dataset = customDataset.customDatasetClass('../test_data')
    test_loader = DataLoader(dataset=test_dataset,batch_size=20,shuffle=False,num_workers=2)
    print('Got the dataloader')
    model = ConvNet()  # Get the model
    print('Got the model')
    
    global num_epochs
    num_epochs=5
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Choose the optimizer and the set the learning rate
    for epoch in range(1,num_epochs+1):
        print('Epoch ', epoch)
        train(model, use_cuda, train_loader, optimizer, epoch)  # Train the network
        test(model, use_cuda, test_loader)  # Test the network
    saveModel(model)


def saveModel(model):
    torch.save(model.state_dict(), f'savedmodel/final_model{num_epochs}.pt')
    # save model in onnx format
    inp = torch.randn(1, 1, 96, 96)
    torch.onnx.export(model, inp, f'savedmodel/model_final{num_epochs}.onnx', verbose=True,
                      input_names=['data'], output_names=['output'])


if __name__ == '__main__':
    main()
