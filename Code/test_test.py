# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:24:01 2017

@author: s160159
"""

## import ----

import os
import re
import sys
import time
import collections

sys.path.append(os.path.join('.', '..')) 
import utils
import utils_s160159 as u_s

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import confusion_matrix

## load data ----
VERSION = '2.0'
FILENAME = 'master'
data_dir = './../Data'
logs_path = './logs'
NUM_SUBJECTS = 1
NUM_CLASSES = 6
VAL_TRAIN_ID = NUM_SUBJECTS - 4

# load all subjects into memory
subjects_list = []
## Load
for ii in range(1,NUM_SUBJECTS+1):
    tmp = np.load(data_dir + '_dicts' + '/subject_' + str(ii) + '_dict.npy').item()
    
    tmp_one = np.zeros((len(tmp[1]),NUM_CLASSES))
    #tmp_one[:] = -1
    for jj in range(len(tmp[1])):
        tmp_one[jj][tmp[1][jj]] = 1
    
    subjects_list.append([tmp[0], tmp[1]])
    break

print('one')
print(dict(collections.Counter(subjects_list[0][1])))
print('hiot')
print(dict(collections.Counter([np.argmax(tmp_one[ii]) for ii in range(len(tmp_one))])))
