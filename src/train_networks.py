import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import os

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from datasets import ImageDataset
from models import VGG16, ResNet

batch_size = 64

class AddGaussianBlur(object):
    def __init__(self, kernel_range=(3, 10)):
        self.kernel_range = kernel_range
        self.kernel_size = 3
        
    def __call__(self, tensor):
        seed = np.random.rand()
        if (seed<0.5):
            self.kernel_size = int(((self.kernel_range[1]-self.kernel_range[0])*np.random.rand()+self.kernel_range[0])/2.0)*2+1
            return transforms.Compose([transforms.GaussianBlur(kernel_size=self.kernel_size)])(tensor)
        else:
            return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(kernel_size={0})'.format(self.kernel_size)

class AdjustBrightness(object):
    def __init__(self, brightness_range=(-1.0, 1.0)):
        self.brightness_range = brightness_range
        self.brightness_factor = 1.0
        
    def __call__(self, tensor):
        seed = np.random.rand()
        if (seed<0.5):
            self.brightness_factor = 10**((self.brightness_range[1]-self.brightness_range[0])*np.random.rand()+self.brightness_range[0])
            return transforms.functional.adjust_brightness(tensor,brightness_factor=self.brightness_factor)
        else:
            return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(brightness_factor={0})'.format(self.brightness_factor)


transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((224, 224)),
    AddGaussianBlur(),
    AdjustBrightness()
    ])

trainloader = torch.utils.data.DataLoader(ImageDataset('data/train.csv', 'data', transform=transform), batch_size=batch_size,
                                        shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(ImageDataset('data/test.csv', 'data', transform=transform), batch_size=batch_size,
                                        shuffle=True, num_workers=2)


if __name__ == "__main__":
        
    net = VGG16(num_classes=9)

    device = torch.device("cuda:0")

    if (not os.path.exists('./model/')):
        os.mkdir('model')

    PATH = './model/model.pth'
    net.to(device)
    # net.load_state_dict(torch.load(PATH))
    # net.eval()
    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
    ep = 30

    total_step = len(trainloader)

    for epoch in range(ep):
        for i, (images, labels) in enumerate(trainloader):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, ep, i+1, total_step, loss.item()))
                
        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in testloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs
            print('Accuracy of the network on the {} validation images: {} %'.format(total, 100 * correct / total)) 

    print('Finished Training')