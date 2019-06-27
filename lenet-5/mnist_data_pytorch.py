import torch
import torchvision.utils as utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.multiprocessing import freeze_support
import matplotlib.pyplot as plt
import PIL
import numpy as np
import random
from lenet5 import Net

def load_data():
    composed_transforms = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    train_dataset = dset.MNIST(root='mnist',transform=composed_transforms,\
                                    train=True, download=True)
    test_dataset = dset.MNIST(root='mnist',transform=composed_transforms,\
                                    train=False, download=True)
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    testloader = DataLoader(test_dataset, batch_size = 64, shuffle=True, num_workers = 4)
    return trainloader, testloader

def inspect_images(dataloader):
    # Look at first 5 images of the first (shuffled) batch
    dataIter = iter(dataloader)
    batch = next(dataIter)
    images_sub_list = [batch[0][i] for i in range(0,5)]
    labels_sub_list = [int(batch[1][i]) for i in range(0,5)]
    grid = utils.make_grid(images_sub_list)
    plt.imshow(grid.numpy().transpose((1,2,0)))
    plt.title('Batch from dataloader')
    # plt.axis('off')
    plt.xticks([32*i + 32/2 for i in range(0,5)], labels_sub_list)
    plt.show()

if __name__ == '__main__':
    load_data()
