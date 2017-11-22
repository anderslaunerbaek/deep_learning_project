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

sys.path.append(os.path.join('.', '..')) 
import utils
import utils_s160159 as u_s

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import confusion_matrix


## load data ----
VERSION = '1.1'
FILENAME = 'master'
data_dir = './../Data'
logs_path = './logs'
NUM_SUBJECTS = 20
NUM_CLASSES = 6
VAL_TRAIN_ID = NUM_SUBJECTS - 4

# load all subjects into memory
subjects_list = []
for i in range(1,NUM_SUBJECTS+1):
    print("Loading subject %d of %d..." %(i, NUM_SUBJECTS), end='\r')
    inputs_night1, targets_night1, _  = u_s.load_spectrograms(data_path=data_dir, 
                                                              subject_id=i, 
                                                              night_id=1,
                                                             no_class=NUM_CLASSES)
    if i!=20:
        inputs_night2, targets_night2, _  = u_s.load_spectrograms(data_path=data_dir, 
                                                                  subject_id=i, 
                                                                  night_id=2,
                                                             no_class=NUM_CLASSES)
    else:
        inputs_night2 = np.empty((0,224,224,3),dtype='uint8')
        targets_night2 = np.empty((0,NUM_CLASSES),dtype='uint8')           

    current_inputs = np.concatenate((inputs_night1,inputs_night2),axis=0)
    current_targets = np.concatenate((targets_night1, targets_night2),axis=0)    
    subjects_list.append([current_inputs, current_targets])       
# extract image shapes
IMAGE_SHAPE = subjects_list[0][0].shape

print(subjects_list)

