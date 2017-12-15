
# coding: utf-8

# # sen_map notebook

# In[ ]:

from __future__ import absolute_import, division, print_function 
import matplotlib
matplotlib.use('Agg')
#import os
#import re
#import sys
#import time

#sys.path.append(os.path.join('.', '..')) 
#import utils
#import utils_DL
import utils_s160159 as u_s

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# ## Load data

# In[ ]:

data_dir = './../Data'
logs_path = './logs'
save_dir = './../Written work/Article/pics/'
SUBJECT_NO = [18,19]
NUM_CLASSES = 6
BATCH_SIZE = 32


# In[ ]:

#print("Loading subject %d..." %(SUBJECT_NO[0]))
#subjects_list_1 = np.load(data_dir + '_dicts' + '/subject_' + str(SUBJECT_NO[0]) + '_dict.npy').item()
#print("Loading subject %d..." %(SUBJECT_NO[1]))
#subjects_list_2 = np.load(data_dir + '_dicts' + '/subject_' + str(SUBJECT_NO[1]) + '_dict.npy').item()
## to dict
#subjects_list={}
#subjects_list[0] = np.concatenate((subjects_list_1[0], subjects_list_2[0]),axis=0)
#subjects_list[1] = np.concatenate((subjects_list_1[1], subjects_list_2[1]),axis=0)
#np.save(data_dir + '_dicts/subject_val_dict', subjects_list)
subjects_list = np.load(data_dir + '_dicts/subject_val_dict.npy').item()

IMAGE_SHAPE = subjects_list[0][0].shape


# In[ ]:

# select first element which forefills this..
state_W = np.where(subjects_list[1] == 0)[0][0]
state_N1 = np.where(subjects_list[1] == 1)[0][0]
state_N2 = np.where(subjects_list[1] == 2)[0][0]
state_N3 = np.where(subjects_list[1] == 3)[0][0]
state_N4 = np.where(subjects_list[1] == 4)[0][0]
state_R = np.where(subjects_list[1] == 5)[0][0]

idx = [[state_W],[state_N1],[state_N2],[state_N3],[state_N4],[state_R]]


# In[ ]:

# input pic
for j in range(len(idx)):
    sub_list_avg = np.empty((0,224,224,3))
    print("Printing sleep stage %d..." %(j))
    for i in idx[j]:
        sub_list_avg=np.concatenate((sub_list_avg, [subjects_list[0][i]]), axis=0) 
    
    #
    # Calcualte Sensitivity map for each class
    sm = np.mean(np.abs(sub_list_avg), axis=0)
    #Scale between 0 and 1
    sm_min=np.min(sm)
    sm_max=np.max(sm)
    sensitivity_map = (sm-sm_min)/(sm_max-sm_min)
    u_s.cal_sen_map(grad_accum=sensitivity_map,
                save_dir=save_dir,
                IMAGE_SHAPE=IMAGE_SHAPE, 
                sen_map_class='clean_' + str(subjects_list[1][i]))


# ## Restoring the master model and  create per class sensitivity map

# In[ ]:

# select first element which forefills this..
state_W = np.where(subjects_list[1] == 0)[0]
state_N1 = np.where(subjects_list[1] == 1)[0]
state_N2 = np.where(subjects_list[1] == 2)[0]
state_N3 = np.where(subjects_list[1] == 3)[0]
state_N4 = np.where(subjects_list[1] == 4)[0]
state_R = np.where(subjects_list[1] == 5)[0]

idx = [state_W,state_N1,state_N2,state_N3,state_N4,state_R]


# In[ ]:

tf.reset_default_graph()
model_path = './models/master/Version_4.0/'
with tf.Session() as sess:
    try:
        # restore model   
        #First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph(model_path + 'fold_1.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        graph = tf.get_default_graph()
        
        x_pl = graph.get_tensor_by_name(name='input_placeholder:0')
        y_pl = graph.get_tensor_by_name(name='target_placeholder:0')
        cross_entropy = graph.get_tensor_by_name('cross_entropy:0')
        sen_map = tf.gradients(cross_entropy, [x_pl], name='sen_map')[0]
        
        #
        for j in range(len(idx)):
            #
            sub_list_avg = np.empty((0,224,224,3))
            # one hot
            tmp = np.zeros((1,NUM_CLASSES))
            tmp[:,j] = 1
            print("Printing sleep stage %d..." %(j))
            for i in idx[j]:
                
                # compute grads
                ce = sess.run(fetches=sen_map,
                         feed_dict={x_pl: [subjects_list[0][i]],
                                    y_pl: tmp})
                #
                sub_list_avg = np.concatenate((sub_list_avg, ce), axis=0)
            
            #
            # Calcualte Sensitivity map for each class
            sm = np.mean(np.abs(sub_list_avg), axis=0)
            #Scale between 0 and 1
            sm_min=np.min(sm)
            sm_max=np.max(sm)
            sensitivity_map = (sm-sm_min)/(sm_max-sm_min)
            u_s.cal_sen_map(grad_accum=sensitivity_map,
                        save_dir=save_dir,
                        IMAGE_SHAPE=IMAGE_SHAPE, 
                        sen_map_class='master_' + str(subjects_list[1][i]))
                
        # close session
        sess.close()

    except KeyboardInterrupt:
        pass


# In[ ]:

tf.reset_default_graph()
model_path = './models/rnn/Version_4.0/'
with tf.Session() as sess:
    try:
        # restore model   
        #First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph(model_path + 'fold_1.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        graph = tf.get_default_graph()
        
        x_pl = graph.get_tensor_by_name(name='input_placeholder:0')
        y_pl = graph.get_tensor_by_name(name='target_placeholder:0')
        cross_entropy = graph.get_tensor_by_name('cross_entropy:0')
        sen_map = tf.gradients(cross_entropy, [x_pl], name='sen_map')[0]
        
        #
        for j in range(len(idx)):
            #
            sub_list_avg = np.empty((0,224,224,3))
            # one hot
            tmp = np.zeros((BATCH_SIZE, NUM_CLASSES))
            tmp[:,j] = 1
            
            print("Printing sleep stage %d..." %(j))
            for i in idx[j]:
                
                #
                tmp_pic = np.zeros((BATCH_SIZE, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]))
                tmp_pic[:] = subjects_list[0][i]
                
                # compute grads
                ce = sess.run(fetches=sen_map,
                         feed_dict={x_pl: tmp_pic,
                                    y_pl: tmp})
                
                #
                sub_list_avg = np.concatenate((sub_list_avg, ce[0:1]), axis=0)
                
            #
            # Calcualte Sensitivity map for each class
            sm = np.mean(np.abs(sub_list_avg), axis=0)
            #Scale between 0 and 1
            sm_min=np.min(sm)
            sm_max=np.max(sm)
            sensitivity_map = (sm-sm_min)/(sm_max-sm_min)
            u_s.cal_sen_map(grad_accum=sensitivity_map,
                        save_dir=save_dir,
                        IMAGE_SHAPE=IMAGE_SHAPE, 
                        sen_map_class='rnn_' + str(subjects_list[1][i]))
                
        # close session
        sess.close()

    except KeyboardInterrupt:
        pass


# In[ ]:



