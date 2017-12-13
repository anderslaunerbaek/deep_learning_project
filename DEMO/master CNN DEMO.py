
# coding: utf-8

# # DEMO
# 
# Learn subject 7 first night
# predict sunject 7 second night

# In[ ]:


"""
Created on Tue Dec 12 22:01:01 2017

@author: s160159
"""

"""
IMPORT LIBRARIES
- 
"""

import os
import re
import sys
import time

sys.path.append(os.path.join('.', '..')) 
import sys
sys.path.insert(0, './../Code')

import utils
import utils_s160159 as u_s
import vgg
# import lstm # used for second model

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import confusion_matrix
import requests


# In[ ]:

"""
LOAD DATA
- 
"""
# download files
if not os.path.isfile('./vgg_16.ckpt'):
    url = 'https://dtudk-my.sharepoint.com/personal/s160159_win_dtu_dk/_layouts/15/guestaccess.aspx?docid=02eb40877b0de45b59cf982888a40b0cd&authkey=AW9R2Yzf5g47BkgK7C35yQs&e=21160da8c61d4293854c120cd915bf4b'
    r = requests.get(url, allow_redirects=True)
    open('./vgg_16.ckpt', 'wb').write(r.content)
if not os.path.isfile('./subject_7_dict.npy'):
    url = 'https://dtudk-my.sharepoint.com/personal/s160159_win_dtu_dk/_layouts/15/guestaccess.aspx?docid=01ae09a062ffe4e35b520f0e2b5253a47&authkey=AV6Ut7L4lzUTQp9EJoHYr2s&e=9590390d0fa24a5ea761226b213243b1'
    r = requests.get(url, allow_redirects=True)
    open('./subject_7_dict.npy', 'wb').write(r.content)

VERSION = 'demo'
FILENAME = 'CNN'
TRAIN_MODEL = True
INITIAL_MODEL_PATH ='./vgg_16.ckpt'
model_path = './Model/'
NUM_CLASSES = 6
NUM_SUBJECT = 7
if not os.path.isfile(model_path + 'demo.ckpt'):
    print("You have to train the model")
    TRAIN_MODEL = True
    os.makedirs(model_path,exist_ok=True)


# In[ ]:

# Load subject into memory

print("Loading subject %d..." %(NUM_SUBJECT))
tmp = np.load('subject_' + str(NUM_SUBJECT) + '_dict.npy').item()

tmp_one = np.zeros((len(tmp[1]),NUM_CLASSES))
#tmp_one[:] = -1
for jj in range(len(tmp[1])):
    tmp_one[jj][tmp[1][jj]] = 1

subjects_list_pic = tmp[0]
subjects_list_target = tmp_one
# no. obs
N = subjects_list_pic.shape[0]
print('Shape pics')
print(subjects_list_pic.shape)
print('Shape targets')
print(subjects_list_target.shape)


# In[ ]:

"""
HYPERPARAMETERS:
- 
"""
# extract image shapes
IMAGE_SHAPE = subjects_list_pic.shape
# hyperameters
HEIGTH, WIDTH, NCHANNELS = IMAGE_SHAPE[1], IMAGE_SHAPE[2], IMAGE_SHAPE[3]
L_RATE = 10e-5
L_RATE_MO_1 = 0.9
L_RATE_MO_2 = 0.999
EPS = 1e-8
DO_KEEP_PROB = 0.5

# Training Loop
MAX_EPOCHS = 20
BATCH_SIZE = 32

# GPU configs
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction = 0.95


# In[ ]:

"""
BUILD THE MODEL:
- heavily inspired by:
 - https://github.com/huyng/tensorflow-vgg/blob/master/layers.py
 - https://raw.githubusercontent.com/tensorflow/models/master/research/slim/nets/vgg.py
"""
# init model
tf.reset_default_graph()

# init placeholders
x_pl = tf.placeholder(tf.float32, [None, HEIGTH, WIDTH, NCHANNELS], name='input_placeholder')
y_pl = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='target_placeholder')

# init pretained TF graph 
logits, _ = vgg.vgg_16(inputs=x_pl, 
                       dropout_keep_prob=DO_KEEP_PROB, 
                       spatial_squeeze=True, 
                       is_training=TRAIN_MODEL, 
                       num_classes=NUM_CLASSES)

# restart from checkpoint
init_fn = slim.assign_from_checkpoint_fn(ignore_missing_vars=True,
                                         model_path=INITIAL_MODEL_PATH, 
                                         var_list=slim.get_variables_to_restore(exclude=['vgg_16/fc6',
                                                                                         'vgg_16/fc7',
                                                                                         'vgg_16/fc8']))



# In[ ]:

"""
TRAIN MODEL
"""
probs = tf.nn.softmax(logits)
prediction = tf.one_hot(tf.argmax(probs, axis=1), depth=NUM_CLASSES)
prediction_bool = tf.equal(tf.argmax(probs, axis=1), tf.argmax(y_pl, axis=1))
accuracy = tf.reduce_mean(tf.cast(prediction_bool, tf.float32))

# computing cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=y_pl,
                                                        name='cross_entropy'))
# defining our optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=L_RATE,  
                                   beta1=L_RATE_MO_1, 
                                   beta2=L_RATE_MO_2, 
                                   epsilon = EPS)
# applying the gradients
train_model = optimizer.minimize(cross_entropy)

# https://stackoverflow.com/questions/35226428/how-do-i-get-the-gradient-of-the-loss-at-a-tensorflow-variable
# https://www.tensorflow.org/versions/r0.12/api_docs/python/train/gradient_computation
sensitivity_map = tf.gradients(cross_entropy, [x_pl], name='sensitivity_map')[0]


# In[ ]:

"""
TESTING THE MODEL STRUCTURE 
"""
sess = tf.Session(config=config)
with sess.as_default():    
    x_batch_1 = subjects_list_pic[0:1]
    y_batch_1 = subjects_list_target[0:1]
    if TRAIN_MODEL:
        print("Loading Initial model...")
        sess.run(tf.global_variables_initializer())
        init_fn(sess)
        print("Initial model loaded...")
    else:
        print("Loading trained model from path:" + model_path + "...")
        saver = tf.train.import_meta_graph(model_path + 'demo.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        graph = tf.get_default_graph()
        
        x_pl = graph.get_tensor_by_name(name='input_placeholder:0')
        y_pl = graph.get_tensor_by_name(name='target_placeholder:0')
        probs = graph.get_tensor_by_name('Softmax:0')
        prediction = tf.one_hot(tf.argmax(probs, axis=1), depth=NUM_CLASSES)
        print("Trained model from path:" + model_path + " loaded...")
        #
    """
    Testing forward pass
    """
    logits, pred, loss, acc = sess.run(fetches=[logits, prediction, cross_entropy, accuracy],
                      feed_dict={x_pl: x_batch_1,
                                 y_pl: y_batch_1})
    assert logits.shape == (1, 6)
    assert pred.shape == (1, 6)
    assert loss >= 0
    assert acc >= 0
    #
    sess.close()


# In[ ]:

"""
Split into test an train
"""
loo = KFold(n_splits=2)
fold = 1   
for idx_train, idx_test in loo.split(list(range(len(subjects_list_pic)))):
    print("Fold %d of %d" %(fold, loo.n_splits))

    # INTO TRAIN 
    inputs_train_ep = np.empty((0,224,224,3),dtype='uint8')  
    targets_train_ep = np.empty((0,NUM_CLASSES),dtype='uint8') 
    for ii in idx_train:
        inputs_train_ep = np.concatenate((inputs_train_ep, [subjects_list_pic[ii]]),axis=0)
        targets_train_ep = np.concatenate((targets_train_ep, [subjects_list_target[ii]]),axis=0)
    #INTO TEST
    inputs_test = np.empty((0,224,224,3),dtype='uint8')  
    targets_test = np.empty((0,NUM_CLASSES),dtype='uint8') 
    for ii in idx_train:
        inputs_test = np.concatenate((inputs_test, [subjects_list_pic[ii]]),axis=0)
        targets_test = np.concatenate((targets_test, [subjects_list_target[ii]]),axis=0)
    
    break # only one permulation


# In[ ]:

"""
TRAIN MODEL
"""
capture_dict = {}
sess = tf.Session(config=config)
with sess.as_default():
    try:
        print('Begin training loop... \n')
        print("Loading Initial pretrained model...")
        sess.run(tf.global_variables_initializer())
        init_fn(sess)
        
        """
        TRAIN
        """
        # valid_loss, valid_accuracy = [], []
        train_loss, train_accuracy = [], []
        test_loss, test_accuracy = [], []

        # LOOP EPOCHS
        print('\tTrain model')
        for epoch in range(MAX_EPOCHS):
            print('\tEpoch: ' + str(epoch + 1) + ' of ' + str(MAX_EPOCHS))
            # down sample
            inputs_train, targets_train = u_s.down_sample(inputs_=inputs_train_ep, 
                                                          targets_=targets_train_ep, 
                                                          no_class=NUM_CLASSES)
            
            max_mini_batch = np.ceil(1+len(inputs_train)/BATCH_SIZE)
            _iter = 1
            for x_batch, y_batch in utils.iterate_minibatches(batchsize=BATCH_SIZE, 
                                                              inputs=inputs_train, 
                                                              targets=targets_train, 
                                                              shuffle=True):
                #

                _, _loss, _acc = sess.run(fetches=[train_model,cross_entropy,accuracy],
                                          feed_dict={x_pl: x_batch,
                                                     y_pl: y_batch})
                #
                print("\t\tminibatch: %d ~ %d\tLOSS: %f\tACCs: %f" %(_iter,max_mini_batch,_loss,_acc),end='\r')
                _iter += 1
                # end loop
            print('')
            # end loop

        """
        get TEST confusion matrix, same approach with validation...
        """
        print('')
        print('\tTest model')
        _iter = 1
        test_pred, test_pred_y_batch = [], []
        max_mini_batch = np.ceil(1+len(inputs_test)/BATCH_SIZE)
        #
        for x_batch, y_batch in utils.iterate_minibatches(batchsize=BATCH_SIZE, 
                                                          inputs=inputs_test, 
                                                          targets=targets_test, 
                                                          shuffle=False):
            _pred = sess.run(fetches=prediction,
                             feed_dict={x_pl: x_batch,
                                        y_pl: y_batch})
            # append prediction
            test_pred += [np.argmax(_pred[ii]) for ii in range(len(_pred))]
            test_pred_y_batch += [np.argmax(y_batch[ii]) for ii in range(len(y_batch))]
            print("\t\tminibatch: %d ~ %d" %(_iter,max_mini_batch),end='\r')
            _iter += 1
            # end loop
            
        # calculate performance
        cm_test = confusion_matrix(y_pred=test_pred, 
                                  y_true=test_pred_y_batch, 
                                  labels=list(range(NUM_CLASSES)))
        print('')
        # SAVE TF model for demo
        save_path = tf.train.Saver().save(sess, model_path + '/demo.ckpt')
        print("Model saved in file: %s" %(model_path))

        # close session
        sess.close()

    except KeyboardInterrupt:
        pass
            


# ## Load trained model and create your confusion matrix

# In[ ]:

tf.reset_default_graph()
with tf.Session() as sess:
    try:
        # restore model   
        # First let's load meta graph 
        saver = tf.train.import_meta_graph(model_path + 'demo.ckpt.meta')
        # ... and restore weights
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        graph = tf.get_default_graph()
        # recall needed tensors
        x_pl = graph.get_tensor_by_name(name='input_placeholder:0')
        y_pl = graph.get_tensor_by_name(name='target_placeholder:0')
        probs = graph.get_tensor_by_name('Softmax:0')
        prediction = tf.one_hot(tf.argmax(probs, axis=1), depth=NUM_CLASSES)
        
        print('')
        print('\tRe-create confusion matric from trained model...')
        _iter = 1
        test_pred, test_pred_y_batch = [], []
        max_mini_batch = np.ceil(1+len(inputs_test)/BATCH_SIZE)
        #
        for x_batch, y_batch in utils.iterate_minibatches(batchsize=BATCH_SIZE, 
                                                          inputs=inputs_test, 
                                                          targets=targets_test, 
                                                          shuffle=False):
            _pred = sess.run(fetches=prediction,
                             feed_dict={x_pl: x_batch,
                                        y_pl: y_batch})
            # append prediction
            test_pred += [np.argmax(_pred[ii]) for ii in range(len(_pred))]
            test_pred_y_batch += [np.argmax(y_batch[ii]) for ii in range(len(y_batch))]

            print("\t\tminibatch: %d ~ %d" %(_iter,max_mini_batch),end='\r')
            _iter += 1
            # end loop
            
        # calculate performance
        cm_test_re = confusion_matrix(y_pred=test_pred,
                                      y_true=test_pred_y_batch, 
                                      labels=list(range(NUM_CLASSES)))
        print('')
      
        # close session
        sess.close()
        #
    except KeyboardInterrupt:
        pass


# ## Compare confusions matrices
# 
# I expect more less same results.

# In[ ]:

print('From traning session')
print(cm_test)
_, _, precision, recall, F1, Acc = u_s.performance_measure(cm_test)
print('--------------------------------------------')
print('Average for all classes')
print('Precision: %f' %(np.mean(precision)))
print('Recall:    %f' %(np.mean(recall)))
print('F1:        %f' %(np.mean(F1)))
print('Acc:       %f' %(np.mean(Acc)))

print('From reconstruction session')
print(cm_test_re)
_, _, precision, recall, F1, Acc = u_s.performance_measure(cm_test_re)
print('--------------------------------------------')
print('Average for all classes')
print('Precision: %f' %(np.mean(precision)))
print('Recall:    %f' %(np.mean(recall)))
print('F1:        %f' %(np.mean(F1)))
print('Acc:       %f' %(np.mean(Acc)))


# In[ ]:



