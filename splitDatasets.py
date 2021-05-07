#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 11:50:33 2021

@author: zhi
"""

import os
from random import seed
from random import sample

seq = range(0, 4200)
randIdx = sample(seq, 1000)

dataFolder = '/home/zhi/projects/faultDiagnosis/phm/class0_28_50hz_High/'
splitFolder = '/home/zhi/projects/faultDiagnosis/phm/class0_28_50hz_High_3200_end/'
os.mkdir(splitFolder)
dataList = sorted(os.listdir(dataFolder))
os.chdir(dataFolder)

for d in dataList:
    dataIdx = d.split('.')[0]
    dataIdx = int(dataIdx.split('__')[-1])
    if dataIdx in randIdx:
        os.rename(d, splitFolder + d)