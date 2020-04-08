import numpy as np
import torch
import torchvision


def fetch_dataloader(mode, transform, batch_size=16):

    train_set = torchvision.datasets.CIFAR10(
        root='./data-cifar-10', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.CIFAR10(
        root='./data-cifar-10', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False)

    if mode == "train":
        return train_loader
    else:
        return test_loader
