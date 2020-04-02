import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def fetch_dataloader(mode, augmentation=False, batch_size=16):
    if augmentation:
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    test_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    train_set = torchvision.datasets.CIFAR10(
        root='./data-cifar-10', train=True, download=True, transform=train_transformer)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.CIFAR10(
        root='./data-cifar-10', train=False, download=True, transform=test_transformer)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False)

    if mode == "train":
        return train_loader
    else:
        return test_loader
