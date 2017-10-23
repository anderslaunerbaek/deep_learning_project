# -*- coding: utf-8 -*-
"""
Created on 09 10 2017

@author: s160159
"""

import numpy as np
from skimage import io
from scipy.misc import imread, imresize


# eeg_vgg_sleep_age_5classes.py
def load_spectrograms(data_path, subject_id, night_id):
    NUM_CLASSES = 6
    sensors='fpz'
    labels = np.loadtxt(data_path+'/sub'+str(subject_id)+'_n'+str(night_id)+'_img_'+sensors+'/labels.txt', dtype='str')

    #
    num_images = np.size(labels)
    targets = np.zeros((num_images), dtype='uint8')
    targets[:]=-1    
    targets[labels=="b'W'"] = 0
    targets[labels=="b'1'"] = 1
    targets[labels=="b'2'"] = 2
    targets[labels=="b'3'"] = 3
    targets[labels=="b'4'"] = 4
    targets[labels=="b'R'"] = 5

    targets = targets[targets!=-1]
    num_images = np.size(targets)

    # one hot
    # if greater then zero
    targets[targets > NUM_CLASSES - 1] = 0
    targets_one_hot = np.zeros((len(targets), NUM_CLASSES))
    targets_one_hot[np.arange(len(targets)), targets] = 1

    # init 
    inputs = np.zeros((num_images,224,224,3),dtype='uint8')

    for idx in range(1,num_images+1):    
        rawim = io.imread(data_path+'/sub'+str(subject_id)+'_n'+str(night_id)+'_img_'+sensors+'/img_'+ np.str(idx) +'.png')
        rawim = rawim[:,:,0:3]

        h, w, _ = rawim.shape
        if not (h==224 and w==224):
        	rawim = skimage.transform.resize(rawim, (224, 224), preserve_range=True)
        #
        inputs[idx-1,:,:,:]=rawim
    #
    return inputs, targets_one_hot, targets
