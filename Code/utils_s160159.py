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

    #
    #inputs = np.zeros((num_images,3,224,224),dtype='uint8')
    inputs = np.zeros((num_images,224,224,3),dtype='uint8')

    for idx in range(1,num_images+1):    
        rawim = io.imread(data_path+'/sub'+str(subject_id)+'_n'+str(night_id)+'_img_'+sensors+'/img_'+ np.str(idx) +'.png')
        rawim = rawim[:,:,0:3]

        h, w, _ = rawim.shape
        if not (h==224 and w==224):
        	rawim = skimage.transform.resize(rawim, (224, 224), preserve_range=True)

        # Shuffle axes to c01
        #im = np.transpose(rawim,(2,0,1))
        im = np.transpose(rawim,(0,1,2))
        im = im[np.newaxis]
        # img1 = imread(data_path+'/sub'+str(subject_id)+'_n'+str(night_id)+'_img_'+sensors+'/img_'+ np.str(idx) +'.png', mode='RGB')
        # img1 = imresize(img1, (224, 224))

        inputs[idx-1,:,:,:]=im
        # inputs[idx-1,:,:,:]=img1

    #
    return inputs, targets