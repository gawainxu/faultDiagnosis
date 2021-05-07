#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 00:31:17 2020

@author: zhi
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from Datasets import ImageTSDataset


'''
This code is utilized to classify the bearing faults 
'''


class LeNet_5(nn.Module):
    
    def __init__(self, inDim, outDim):
        super(LeNet_5, self).__init__()
        '''
        outDim is the number of classes
        '''
        self.inDim = inDim
        self.outDim = outDim
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=5, padding=(2,2))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=5,  padding=(2,2))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=25, kernel_size=5, padding=(2,2))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.linear1 = nn.Linear(inDim*inDim/4/4/4*25, 1000)     #2000
        self.linear2 = nn.Linear(1000, 500)
        self.outlayer = nn.Linear(500, outDim)
        
        self.dropout = nn.Dropout(p=0.5)
        
        
    def forward(self, x):
        
        y = F.relu(self.conv1(x))
        y = self.pool1(y)
        y = F.relu(self.conv2(y))
        y = self.pool2(y)
        y = F.relu(self.conv3(y))
        y = self.pool3(y)
        
        y = torch.flatten(y, start_dim = 1)
        
        y = F.relu((self.linear1(y)))
        y = F.relu((self.linear2(y)))
        y = self.outlayer(y)
        #y = F.log_softmax((y))
        
        return y
    
    
    
class LeNet_enhanced(nn.Module):
    
    def __init__(self, inDim, outDim):
        super(LeNet_enhanced, self).__init__()
        '''
        outDim is the number of classes
        '''
        self.inDim = inDim
        self.outDim = outDim
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=5, padding=(2,2))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=5,  padding=(2,2))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=5, padding=(2,2))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=25, kernel_size=5, padding=(2,2))
        
        self.linear1 = nn.Linear(inDim*inDim/4/4/4*25, 1000)     
        self.linear2 = nn.Linear(1000, 500)
        self.outlayer = nn.Linear(500, outDim)
        
        self.dropout = nn.Dropout(p=0.5)
        
        
    def forward(self, x):
        
        y = F.relu(self.conv1(x))
        y = self.pool1(y)
        y = F.relu(self.conv2(y))
        y = self.pool2(y)
        y = F.relu(self.conv3(y))
        y = self.pool3(y)
        y = F.relu(self.conv4(y))
        
        y = torch.flatten(y, start_dim = 1)
        
        y = F.relu((self.linear1(y)))
        y = F.relu((self.linear2(y)))
        y = F.log_softmax((self.outlayer(y)))
        
        return y
    
    


class LeNet_enhanced1(nn.Module):
    
    def __init__(self, inDim, outDim):
        super(LeNet_enhanced1, self).__init__()
        '''
        outDim is the number of classes
        '''
        self.inDim = inDim
        self.outDim = outDim
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=5, padding=(2,2))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=5,  padding=(2,2))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=5, padding=(2,2))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=25, kernel_size=5, padding=(2,2))
        
        self.linear1 = nn.Linear(inDim*inDim/4/4/4*25, 1000)     
        self.linear2 = nn.Linear(1000, 500)
        self.linear3 = nn.Linear(500, 200)
        self.outlayer = nn.Linear(200, outDim)
        
        self.dropout = nn.Dropout(p=0.5)
        
        
    def forward(self, x):
        
        y = F.relu(self.conv1(x))
        y = self.pool1(y)
        y = F.relu(self.conv2(y))
        y = self.pool2(y)
        y = F.relu(self.conv3(y))
        y = self.pool3(y)
        y = F.relu(self.conv4(y))
        
        y = torch.flatten(y, start_dim = 1)
        
        y = F.relu((self.linear1(y)))
        y = F.relu((self.linear2(y)))
        y = F.relu((self.linear3(y)))
        y = F.log_softmax((self.outlayer(y)))
        
        return y
    
    
class LeNet_enhanced2(nn.Module):
    
    def __init__(self, inDim, outDim):
        super(LeNet_enhanced2, self).__init__()
        '''
        outDim is the number of classes
        '''
        self.inDim = inDim
        self.outDim = outDim
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=5, padding=(2,2))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=5,  padding=(2,2))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=5, padding=(2,2))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=25, kernel_size=5, padding=(2,2))
        self.conv5 = nn.Conv2d(in_channels=25, out_channels=25, kernel_size=5, padding=(2,2))
        
        self.linear1 = nn.Linear(inDim*inDim/4/4/4*25, 1000)     
        self.linear2 = nn.Linear(1000, 500)
        self.linear3 = nn.Linear(500, 200)
        self.outlayer = nn.Linear(200, outDim)
        
        self.dropout = nn.Dropout(p=0.5)
        
        
    def forward(self, x):
        
        y = F.relu(self.conv1(x))
        y = self.pool1(y)
        y = F.relu(self.conv2(y))
        y = self.pool2(y)
        y = F.relu(self.conv3(y))
        y = self.pool3(y)
        y = F.relu(self.conv4(y))
        y = F.relu(self.conv5(y))
        
        y = torch.flatten(y, start_dim = 1)
        
        y = F.relu((self.linear1(y)))
        y = F.relu((self.linear2(y)))
        y = F.relu((self.linear3(y)))
        y = F.log_softmax((self.outlayer(y)))
        
        return y



class LeNet_enhanced3(nn.Module):
    
    def __init__(self, inDim, outDim):
        super(LeNet_enhanced3, self).__init__()
        '''
        outDim is the number of classes
        '''
        self.inDim = inDim
        self.outDim = outDim
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=7, padding=(3,3))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=7,  padding=(3,3))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=7, padding=(3,3))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=5, padding=(2,2))
        self.conv5 = nn.Conv2d(in_channels=50, out_channels=25, kernel_size=5, padding=(2,2))
        self.conv6 = nn.Conv2d(in_channels=25, out_channels=25, kernel_size=5, padding=(2,2))
        
        self.linear1 = nn.Linear(inDim*inDim/4/4/4*25, 400)      # 800
        self.linear2 = nn.Linear(400, 200)     #400
        self.linear3 = nn.Linear(200, 100)     #200 
        self.outlayer = nn.Linear(100, outDim)
        
        self.dropout = nn.Dropout(p=0.5)
        
        
    def forward(self, x):
        
        y = F.relu(self.conv1(x))
        y = self.pool1(y)
        y = F.relu(self.conv2(y))
        y = self.pool2(y)
        y = F.relu(self.conv3(y))
        y = self.pool3(y)
        y = F.relu(self.conv4(y))
        y = F.relu(self.conv5(y))
        y = F.relu(self.conv6(y))
        
        y = torch.flatten(y, start_dim = 1)
        
        y = F.relu((self.linear1(y)))
        y = self.dropout(y)
        y = F.relu((self.linear2(y)))
        y = self.dropout(y)
        y = F.relu((self.linear3(y)))
        #y = F.log_softmax((self.outlayer(y)))
        y = self.outlayer(y)
        
        return y


    
    
def train(model, device, dataLoader, optimizer):
    # Hint: (for cross_entropy) https://jbencook.com/cross-entropy-loss-in-pytorch/
    model.train()
    model = model.cuda()
    lossEpoch = 0
    
    for batchIdx, (img, label) in enumerate(dataLoader):
        #img, label = torch.from_numpy(img), torch.from_numpy(label)
        img = img.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        optimizer.zero_grad()
        output = model(img)
        #output = F.log_softmax(output)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()
        
        lossEpoch += loss.cpu()
    return lossEpoch / batchIdx
    


if __name__ == '__main__':
    
    N = 64
    numClasses = 14
    batchSize = 32
    lr = 0.01
    epochs = 100
    
    dataMean = -0.0153
    dataStd = 0.291
    
    # Prepare the dataset
    imageDataFoloder = '/home/zhi/projects/faultDiagnosis/phm/class0_14_45Hz_High'
    imageDT = ImageTSDataset(imageDataFoloder)
    
    transform = transforms.Compose([transforms.ToTensor()])      
    imageDTLoader = DataLoader(imageDT, batch_size = batchSize, shuffle=True, num_workers = 4, drop_last=True)
    
    # Prepare the network
    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet_enhanced2(N, numClasses)
    optimizer = optim.Adadelta(model.parameters(), lr = lr)
    model.load_state_dict(torch.load('/home/zhi/projects/faultDiagnosis/phm/LossFiles/LeNet_enhanced2_class0_14_45hz_High.pt'))
    
    
    # Start the training
    
    lossMin = 20
    for e in range(epochs):
        lossEpoch = train(model, device, imageDTLoader, optimizer)
      #  Loss.append(lossEpoch)
        print('Epoch: ', e, 'Loss: ', lossEpoch)
        if lossEpoch < lossMin:
            torch.save(model.state_dict(), '/home/zhi/projects/faultDiagnosis/phm/LossFiles/LeNet_enhanced2_class0_14_45hz_High.pt')
            lossMin = lossEpoch
