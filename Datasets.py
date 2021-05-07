#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:12:04 2020

@author: zhi
"""


import os
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

import matplotlib.pyplot as plt



class ImageTSDataset(Dataset):
    
    def __init__(self, ImageDataFoloder, transform=None):
        self.ImageDataFolder = ImageDataFoloder
        self.ImageDataList  = os.listdir(ImageDataFoloder)
        
        os.chdir(ImageDataFoloder)
        self.ImageDataList = sorted(os.listdir(ImageDataFoloder))
        
        self.transform = transform
        
        self.labelDict = {"helical 1_50hz_Low_1":0, "helical 1_50hz_Low_2":1, 
                          "helical 2_50hz_Low_1":2, "helical 2_50hz_Low_2":3, 
                          "helical 3_50hz_Low_1":4, "helical 3_50hz_Low_2":5, 
                          "helical 4_50hz_Low_1":6, "helical 4_50hz_Low_2":7,
                          "helical 5_50hz_Low_1":8, "helical 5_50hz_Low_2":9,
                          "helical 6_50hz_Low_1":10, "helical 6_50hz_Low_2":11,
                          "spur 1_50hz_Low_1":12, "spur 1_50hz_Low_2":13,}
        
        self.numClasses = len(self.labelDict)
        
        
    def __getitem__(self, idx):
        
        dataName = self.ImageDataList[idx]
        img = np.load(dataName, allow_pickle=True)
        img = np.expand_dims(img, axis=0)
    
        dataIden = dataName.split('.')[0]
        label = dataIden.split('__')[0]      

        if label in self.labelDict.keys():
            label = self.labelDict[label]
        else:
            label = 100
        
        return img, label
        
        
    def __len__(self):
        
        numImages = len(self.ImageDataList)
        return numImages
    
    
    
class ToTensor(object):

     def __call__(self, sample):
         img = sample['image']
         label = sample['label']
        
         return {'image': torch.from_numpy(img), 'label': torch.from_numpy(label)}
    

if __name__ == '__main__':
    
    ImageDataFoloder = '/home/zhi/projects/faultDiagnosis/phm/class0_14_Low_3200_end'
    DT = ImageTSDataset(ImageDataFoloder)