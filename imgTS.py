#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:50:44 2020

@author: zhi
"""

import os                
import numpy as np
import random
           
N = 64
intervel = 64

labelDict = ["helical 1_50hz_High_1", "helical 1_50hz_High_2", 
             "helical 2_50hz_High_1", "helical 2_50hz_High_2", 
             "helical 3_50hz_High_1", "helical 3_50hz_High_2", 
             "helical 4_50hz_High_1", "helical 4_50hz_High_2",
             "helical 5_50hz_High_1", "helical 5_50hz_High_2",
             "helical 6_50hz_High_1", "helical 6_50hz_High_2",
             
             "spur 1_50hz_High_1", "spur 1_50hz_High_2",
             "spur 2_50hz_High_1", "spur 2_50hz_High_2",
             "spur 3_50hz_High_1", "spur 3_50hz_High_2",
             "spur 4_50hz_High_1", "spur 4_50hz_High_2",
             "spur 5_50hz_High_1", "spur 5_50hz_High_2",
             "spur 6_50hz_High_1", "spur 6_50hz_High_2",
             "spur 7_50hz_High_1", "spur 7_50hz_High_2",
             "spur 8_50hz_High_1", "spur 8_50hz_High_2"]

selectList = random.sample(range(0, 28), 14)


dataFolder = './phm/PHM_Society_2009_Competition_Expanded_txt/50Hz_High'
dataFolderList = sorted(os.listdir(dataFolder))
os.chdir(dataFolder)
distFolder = './phm/class0_28_50hz_High'
os.mkdir(distFolder)


for i, fname in enumerate(dataFolderList):
    if i in selectList:
        continue
    print fname
    txtFile = fname + '/' + fname + '.txt'
    ts = []
    f = open(txtFile, 'r')
    lines = f.readlines()
    for l in lines:
        ws = l.split(' ')
        wsn = [w for w in ws if w != '']
        ts.append(float(wsn[1]))
    dlen = len(ts)
            
    for idx, pointer in enumerate(range(0, dlen, intervel)):
        if pointer + N*N > dlen:
            break
        timeSeries = ts[pointer:pointer+N*N]
        timeSeries = np.array(timeSeries)
        timeSeries = timeSeries.reshape([N, N])
        timeSeries.dump(os.path.join(distFolder, fname + '__' + str(idx)+'.dat'))
            
