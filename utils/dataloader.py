import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

def fetch_dataloader(mode, transform, batch_size=16):

    train_set = torchvision.datasets.CIFAR10(
        root='./data/data-cifar-10', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.CIFAR10(
        root='./data/data-cifar-10', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False)

    if mode == "train":
        return train_loader
    else:
        return test_loader


class MyDataset(Dataset):
    def __init__(self, data, targets, transform=_TRANSFORM):
        self.data = data #.transpose((0,2,3,1))
        self.targets = torch.Tensor(targets)
        self.transform = transform
    
    def __getitem__(self, index):
        y = self.targets[index]
        x = self.data[index]        
        if self.transform:
            x = Image.fromarray(np.uint8(x))
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)