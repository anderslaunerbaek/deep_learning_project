# -*- coding: utf-8 -*-
"""
Created on 09 10 2017

@author: s160159
"""

import numpy as np
from skimage import io
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import tensorflow as tf

def down_sample(inputs_, targets_, no_class, verbose = False):
    class_balance = np.sum(targets_,0)
    n = targets_.shape[0]
    if verbose: print('distribution\n' + str(safe_div(class_balance, n)))
        
    #
    if any(class_balance == 0): return inputs_, targets_

    min_class = np.argmin(class_balance)
    min_samples = np.min(class_balance)
    if verbose: print("Keeps %f pct. of the samples" %((1 - (n - min_samples) / n) * 100))
    
    tmp_train = np.empty((0,224,224,3),dtype='uint8')
    tmp_target = np.empty((0,6),dtype='uint8')
    # loop
    for ii in range(no_class):
        idx = np.argmax(targets_,1) == ii
        if not ii == min_class:
            # down sample
            idx_sample = np.random.choice(range(targets_[idx].shape[0]), 
                                          int(min_samples), 
                                          replace=False)
            #
            tmp_train = np.concatenate((tmp_train,inputs_[idx][idx_sample]),axis=0)
            tmp_target = np.concatenate((tmp_target,targets_[idx][idx_sample]),axis=0)
        else:
            # minority class
            tmp_train = np.concatenate((tmp_train,inputs_[idx]),axis=0)
            tmp_target = np.concatenate((tmp_target,targets_[idx]),axis=0)
        # end loop
    #
    return tmp_train, tmp_target

def save_weights(graph , fpath):
    sess = tf.get_default_session()
    variables = graph.get_collection("variables")
    variable_names = [v.name for v in variables]
    kwargs = dict(zip(variable_names, sess.run(variables)))
    np.savez(fpath, **kwargs)


def load_weights(graph, fpath):
    sess = tf.get_default_session()
    variables = graph.get_collection("variables")
    data = np.load(fpath)
    for v in variables:
        if v.name not in data:
            print("could not load data for variable='%s'" % v.name)
            continue
        print("assigning %s" % v.name)
        sess.run(v.assign(data[v.name]))


def safe_div(x,y):
    return np.divide(x,y,where=y != 0)

def cal_sen_map(grad_accum, sen_map_class, IMAGE_SHAPE, save_dir = './pics/'):
    ## Calcualte Sensitivity maps       
    sm = np.mean(np.abs(grad_accum), axis=0)
    ## Scale between 0 and 1
    sensitivity_map = safe_div((sm-np.min(sm)), (np.max(sm)-np.min(sm)))
    f = plt.figure()
    plt.imshow(sensitivity_map, interpolation=None)
    plt.yticks(np.linspace(start=0, stop=IMAGE_SHAPE[1]-1, num=7), 
               ('30','25','20','15','10','5','0'))
    plt.xticks(np.linspace(start=0, stop=IMAGE_SHAPE[2]-1, num=5), 
               ('t','t + 7.5','t + 15','t + 22.5','t + 30'))
    plt.xlabel('time (s)')
    plt.ylabel('frequency (Hz)')
    f.savefig(save_dir + 'class_' + sen_map_class + '.pdf', bbox_inches='tight')
    plt.show()


# eeg_vgg_sleep_age_5classes.py
def load_spectrograms(data_path, subject_id, night_id, no_class):
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
    targets[targets > no_class - 1] = 0
    targets_one_hot = np.zeros((len(targets), no_class))
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
