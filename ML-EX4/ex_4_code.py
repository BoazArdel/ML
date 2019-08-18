import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset
from itertools import accumulate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

class NNet(nn.Module):
    def __init__(self,image_size):
        super(NNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        #self.bn1 = nn.BatchNorm1d(100)
        #self.bn2 = nn.BatchNorm1d(50)
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        #x = F.dropout(x, training=self.training)
        #x = F.relu(self.bn1(self.fc0(x)))
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        #x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)


def train(optimizer, model, train_loader):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()


def test(model,test_loader,file,ifprint=False):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        if (ifprint):
            file.write(str(pred.item()))
            file.write('\n')
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    return test_loss, correct,100. * correct / len(test_loader.dataset)


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    indices = torch.randperm(sum(lengths))
    splits = []
    for i in range(len(lengths)):
        offset = sum(lengths[:i + 1])
        splits.append(Subset(dataset, indices[offset - lengths[i]:offset]))
    return splits


'''
def _make_dataloaders(train_set, train_size, valid_size, batch_size):
    # Split training into train and validation
    indices = torch.randperm(len(train_set))
    train_indices = indices[:len(indices)-valid_size]
    valid_indices = indices[len(indices)-valid_size:] if valid_size else None

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(train_indices)) 
    if valid_size:
        valid_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(valid_indices))
    else:
        valid_loader = None

    return train_loader, valid_loader
'''


def main():
    file = open("results.txt", 'w+')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    data_loader = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transform),batch_size=1, shuffle=False)

    '''
    num_train = len(train_loader.dataset)
    train_loader, valid_loader = _make_dataloaders(train_loader.dataset, int(num_train*0.8), int(num_train*0.2), batch_size=1)
    
   
    #train_data_set = train_loader.dataset
    num_train = len(train_data_set)
    indices = list(range(num_train))
    split = int(num_train*0.2)

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))
    print(str(len(train_idx)))
    # Contiguous split
    # train_idx, validation_idx = indices[split:], indices[:split]

    ## define our samplers -- we use a SubsetRandomSampler because it will return
    ## a random subset of the split defined by the given indices without replaf
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    print(str(len(train_sampler)))

    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=4, sampler=train_sampler)

    validation_loader = torch.utils.data.DataLoader(train_data_set,
                                                    batch_size=2, sampler=validation_sampler)
    '''

    num_train = len(data_loader)
    loader = random_split(data_loader, [int(num_train * 0.8), int(num_train * 0.2)])
    train_loader = torch.utils.data.DataLoader(loader[0], batch_size=1, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(loader[1], batch_size=1, shuffle=True)
    print(str(len(train_loader)))

    model = NNet(image_size=28 * 28)
    epochs = 5
    lr = 0.005

    optimizer = optim.SGD(model.parameters(), lr=lr)
    '''
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.AdaDelta(model.parameters(), lr=lr)
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    '''

    results_train_loss = []
    results_train_correct = []
    results_train_perc = []
    results_valid_loss = []
    results_valid_correct = []
    results_valid_perc = []

    print('train:')
    for epoch in range(1, epochs):
        train(optimizer, model,train_loader)
        loss, correct, percentage = test(model,train_loader,file)
        results_train_loss.append(loss)
        results_train_correct.append(correct)
        results_train_perc.append(percentage)

    '''
    optimizer2 = optim.RMSprop(model.parameters(), lr=0.1)    
    '''
    print('valid:')
    for epoch in range(1, epochs):
        train(optimizer, model,valid_loader)
        loss, correct, percentage = test(model,valid_loader,file)
        results_valid_loss.append(loss)
        results_valid_correct.append(correct)
        results_valid_perc.append(percentage)


    print('test:')
    loss, correct, percentage = test(model, test_loader,file,True)
    print(loss, correct, percentage)
    file.close()

    t = range(1,epochs)
    plt.interactive(False)
    plt.plot(t, results_train_loss, 'r') # plotting t, a - normal dist
    plt.plot(t,  results_valid_loss, 'b')  # plotting t, b - softmax prob
    red_patch = mpatches.Patch(color='red', label='Train')
    blue_patch = mpatches.Patch(color='blue', label='Validation')
    plt.legend(handles=[red_patch,blue_patch])
    plt.title('Loss for ' + str(epochs) + ' epochs')
    plt.ylabel('Average Loss')
    plt.xlabel('epochs')
    plt.show(block=True)
    plt.interactive(False)
    plt.plot(t, results_train_perc, 'r') # plotting t, a - normal dist
    plt.plot(t,  results_valid_perc, 'b')  # plotting t, b - softmax prob
    red_patch = mpatches.Patch(color='red', label='Train')
    blue_patch = mpatches.Patch(color='blue', label='Validation')
    plt.legend(handles=[red_patch,blue_patch])
    plt.title('Percentage for ' + str(epochs) + ' epochs')
    plt.ylabel('Percentage')
    plt.xlabel('epochs')
    plt.show(block=True)


if __name__ == "__main__":
    main()