# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:24:01 2017

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
import utils
import utils_s160159 as u_s
import vgg
import lstm


import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import confusion_matrix

"""
LOAD DATA
- 
"""
VERSION = '4.0'
FILENAME = 'rnn'
TRAIN_MODEL = True
TRAIN_MODEL_PATH ='...'
INITIAL_MODEL_PATH ='./../Data/vgg_16.ckpt'

data_dir = './../Data'
logs_path = './logs'
NUM_SUBJECTS = 20
NUM_CLASSES = 6
VAL_TRAIN_ID = NUM_SUBJECTS - 2

# load all subjects into memory
subjects_list = []
## Load
for ii in range(1,NUM_SUBJECTS+1):
    print("Loading subject %d of %d..." %(ii, NUM_SUBJECTS), end='\r')
    tmp = np.load(data_dir + '_dicts' + '/subject_' + str(ii) + '_dict.npy').item()
    
    tmp_one = np.zeros((len(tmp[1]),NUM_CLASSES))
    #tmp_one[:] = -1
    for jj in range(len(tmp[1])):
        tmp_one[jj][tmp[1][jj]] = 1
    
    subjects_list.append([tmp[0], tmp_one])
# extract image shapes
IMAGE_SHAPE = subjects_list[0][0].shape

"""
HYPERPARAMETERS:
- 
"""

# hyperameters
HEIGTH, WIDTH, NCHANNELS = IMAGE_SHAPE[1], IMAGE_SHAPE[2], IMAGE_SHAPE[3]
L_RATE = 10e-5
L_RATE_MO_1 = 0.9
L_RATE_MO_2 = 0.999
EPS = 1e-8
DO_KEEP_PROB = 0.5

# Training Loop
MAX_EPOCHS = 20 # 50
BATCH_SIZE = 32

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction = 0.95


"""
BUILD THE MODEL:
- https://github.com/huyng/tensorflow-vgg/blob/master/layers.py
- https://raw.githubusercontent.com/tensorflow/models/master/research/slim/nets/vgg.py
"""
# init model
tf.reset_default_graph()

# init placeholders
x_pl = tf.placeholder(tf.float32, [BATCH_SIZE, HEIGTH, WIDTH, NCHANNELS], name='input_placeholder')
y_pl = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES], name='target_placeholder')

# init pretained TF graph 
logits, _ = vgg.vgg_16(inputs=x_pl, 
                       dropout_keep_prob=DO_KEEP_PROB, 
                       spatial_squeeze=True, 
                       is_training=True, 
                       num_classes=NUM_CLASSES,
                       return_lstm=True)

# LSTM 1
logits, _  = lstm.basic_conv_lstm_cell(inputs=logits,
                                           state=None,
                                           forget_bias=1.0, 
                                           filter_size=logits.get_shape()[1].value,
                                           num_channels=logits.get_shape()[3].value)
## normalize
#logits = tf.contrib.layers.layer_norm(inputs=logits)
## LSTM 2
#logits, LSTM_2 = lstm.basic_conv_lstm_cell(inputs=logits,
#                                           state=LSTM_1,
#                                           forget_bias=1.0, 
#                                           filter_size=logits.get_shape()[1].value,
#                                           num_channels=logits.get_shape()[3].value)
## normalize
#logits = tf.contrib.layers.layer_norm(inputs=logits)
## LSTM 3
#logits, LSTM_3 = lstm.basic_conv_lstm_cell(inputs=logits,
#                                           state=LSTM_2,
#                                           forget_bias=1.0, 
#                                           filter_size=logits.get_shape()[1].value,
#                                           num_channels=logits.get_shape()[3].value)
## normalize
#logits = tf.contrib.layers.layer_norm(inputs=logits)
## LSTM 4
#logits, LSTM_4 = lstm.basic_conv_lstm_cell(inputs=logits,
#                                           state=LSTM_3,
#                                           forget_bias=1.0, 
#                                           filter_size=logits.get_shape()[1].value,
#                                           num_channels=logits.get_shape()[3].value)

# Dropout
logits = slim.dropout(inputs=logits, keep_prob=DO_KEEP_PROB, is_training=True, scope='RNN_dropout')
# Reshape
logits = tf.reshape(logits, [int(BATCH_SIZE), -1])
# Fully connect
logits = slim.layers.fully_connected(inputs=logits, num_outputs=NUM_CLASSES, activation_fn=None)

"""

"""
init_fn = slim.assign_from_checkpoint_fn(ignore_missing_vars=True,
                                         model_path=INITIAL_MODEL_PATH, 
                                         var_list=slim.get_variables_to_restore(exclude=['fully_connected',
                                                                                         'LayerNorm',
                                                                                         'BasicConvLstmCell',
                                                                                         'vgg_16/fc6','vgg_16/fc7',
                                                                                         'vgg_16/fc8']))

#init_fn_trained = slim.assign_from_checkpoint_fn(ignore_missing_vars=False, 
#                                                 var_list=tf.GraphKeys.VARIABLES(),
#                                                 model_path=TRAIN_MODEL_PATH)

# slim.get_model_variables()
#no_train_able = []
#for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
#    no_train_able.append(np.prod(i.shape[:]).value)   # i.name if you want just a name   
#print('Model consits of ', np.sum(no_train_able), 'trainable parameters.')


"""

"""
# with tf.variable_scope('performance'):
probs = tf.nn.softmax(logits)
prediction = tf.one_hot(tf.argmax(probs, axis=1), depth=NUM_CLASSES)
prediction_bool = tf.equal(tf.argmax(probs, axis=1), tf.argmax(y_pl, axis=1))
accuracy = tf.reduce_mean(tf.cast(prediction_bool, tf.float32))

#with tf.variable_scope('loss_function'):
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
    
#with tf.variable_scope('sensitivity_map'):   
# https://stackoverflow.com/questions/35226428/how-do-i-get-the-gradient-of-the-loss-at-a-tensorflow-variable
# https://www.tensorflow.org/versions/r0.12/api_docs/python/train/gradient_computation
sensitivity_map = tf.gradients(cross_entropy, [x_pl], name='sensitivity_map')[0]


"""
TESTING THE MODEL STRUCTURE 
"""
#sess = tf.Session(config=config)
#with sess.as_default():    
#x_batch_1 = subjects_list[0][0]
#y_batch_1 = subjects_list[0][1]
#x_batch_2 = subjects_list[1][0][41:62]
#y_batch_2 = subjects_list[1][1][41:62]
#    if TRAIN_MODEL:
#        print("Loading Initial model...", end="\r")
#        sess.run(tf.global_variables_initializer())
#        init_fn(sess)
#        print("Initial model loaded...")
#    else:
#        print("Loading trained model from path:" + TRAIN_MODEL_PATH + "...", end="\r")
#        sess.run(tf.global_variables_initializer())
#        #init_fn(sess)
#        print("Trained model from path:" + TRAIN_MODEL_PATH + " loaded...")
#        #
#    """
#    Testing
#    """
#    print("model trained")
#    logits, pred, loss, acc = sess.run(fetches=[logits, prediction, cross_entropy, accuracy],
#                      feed_dict={x_pl: x_batch_2,
#                                 y_pl: y_batch_2})
#    print("Logits: " + str(logits))
#    print("pred: " + str(pred))
#    print("pred_cor: " + str(y_batch_2))
#    print("loss: " + str(loss))
#    print("acc: " + str(acc))
#    sess.close()


"""
TRAIN MODEL
"""
capture_dict = {}
sess = tf.Session(config=config)
with sess.as_default():
    try:
        START_TIME = time.ctime()
        MODEL_PATH = "./models/"+ FILENAME + "/Version_" + VERSION + "_" + START_TIME
        if not os.path.exists(MODEL_PATH): os.makedirs(MODEL_PATH)
        print('Begin training loop... \n')
        
        if TRAIN_MODEL:
            print("Loading Initial model...", end="\r")
            sess.run(tf.global_variables_initializer())
            init_fn(sess)
            print("Initial model loaded...")
        else:
            print("Loading trained model from path:" + TRAIN_MODEL_PATH + "...", end="\r")
            sess.run(tf.global_variables_initializer())
            #init_fn(sess)
            print("Trained model from path:" + TRAIN_MODEL_PATH + " loaded...")
        
        # INTO VALIDATION
        idx_val = list(range(VAL_TRAIN_ID, NUM_SUBJECTS))
        val_data = [subjects_list[i] for i in idx_val]
        inputs_val = np.empty((0,224,224,3),dtype='uint8')  
        targets_val = np.empty((0,NUM_CLASSES),dtype='uint8') 
        for ii in range(len(val_data)):
            inputs_val = np.concatenate((inputs_val, val_data[ii][0]),axis=0)
            targets_val = np.concatenate((targets_val, val_data[ii][1]),axis=0)    
        
        # CROSS VALIDATION
        # loo = KFold(n_splits=10)
        loo = LeaveOneOut()
        fold = 1   
        for idx_train, idx_test in loo.split(list(range(VAL_TRAIN_ID))):
            print("Fold %d of %d" %(fold, loo.get_n_splits(list(range(VAL_TRAIN_ID)))))
            capture_dict[fold] = {}
            #
            valid_loss, valid_accuracy = [], []
            train_loss, train_accuracy = [], []
            test_loss, test_accuracy = [], []
            
            #INTO TRAIN 
            train_data = [subjects_list[i] for i in idx_train]
            inputs_train_ep = np.empty((0,224,224,3),dtype='uint8')  
            targets_train_ep = np.empty((0,NUM_CLASSES),dtype='uint8') 
            for ii in range(len(train_data)):
                inputs_train_ep = np.concatenate((inputs_train_ep, train_data[ii][0]),axis=0)
                targets_train_ep = np.concatenate((targets_train_ep, train_data[ii][1]),axis=0)
            
            #INTO TEST 
            test_data = [subjects_list[i] for i in idx_test]
            inputs_test = np.empty((0,224,224,3),dtype='uint8')  
            targets_test = np.empty((0,NUM_CLASSES),dtype='uint8') 
            for ii in range(len(test_data)):
                inputs_test = np.concatenate((inputs_test, test_data[ii][0]),axis=0)
                targets_test = np.concatenate((targets_test, test_data[ii][1]),axis=0)
                        
            # LOOP EPOCHS
            print('\tTrain model')
            for epoch in range(MAX_EPOCHS):
                print('\tEpoch: ' + str(epoch + 1) + ' of ' + str(MAX_EPOCHS))
                # TRAIN
                # down sample
                inputs_train, targets_train = u_s.down_sample(inputs_=inputs_train_ep, 
                                                              targets_=targets_train_ep, 
                                                              no_class=NUM_CLASSES) 

                _train_loss, _train_accuracy = [], []
                _iter = 1
                for x_batch, y_batch in utils.iterate_minibatches(batchsize=BATCH_SIZE, 
                                                                  inputs=inputs_train, 
                                                                  targets=targets_train, 
                                                                  shuffle=True):
                    
                    # LSTM has a static structure...
                    if len(y_batch) != BATCH_SIZE: break
                    # 
                    _,_loss,_acc = sess.run(fetches=[train_model, cross_entropy, accuracy],
                                             feed_dict={x_pl: x_batch, y_pl: y_batch})
                    
                    # append to mini batch
                    _train_loss.append(_loss)
                    _train_accuracy.append(_acc)                    
                    #
                    print("\t\tminibatch: %d\tLOSS: %f\tACCs: %f" %(_iter,_loss,_acc))#,end='\r')
                    _iter += 1
                    # end loop
                # append mean loss and accuracy
                print('')
                train_loss.append(np.nanmean(_train_loss))
                train_accuracy.append(np.nanmean(_train_accuracy))
                # end loop

            # COMPUTE VALIDATION LOSS AND ACCURACY
            print('')
            print('\tEvaluate validation performance')
            val_pred, val_pred_y_batch = [], []
            _iter = 1
            #
            for x_batch, y_batch in utils.iterate_minibatches(batchsize=BATCH_SIZE, 
                                                              inputs=inputs_val, 
                                                              targets=targets_val, 
                                                              shuffle=False):
                # LSTM has a static structure...
                if len(y_batch) != BATCH_SIZE: break
                #
                _loss,_acc,_pred = sess.run(fetches=[cross_entropy, accuracy, prediction],
                                            feed_dict={x_pl: x_batch, y_pl: y_batch})
                # append prediction
                val_pred += [np.argmax(_pred[ii]) for ii in range(len(_pred))]
                val_pred_y_batch += [np.argmax(y_batch[ii]) for ii in range(len(y_batch))]
                # append mean
                valid_loss.append(_loss)
                valid_accuracy.append(_acc)
                print("\t\tminibatch: %d\tLOSS: %f\tACCs: %f" %(_iter,_loss,_acc),end='\r')
                _iter += 1
                # end loop
            # calculate performance
            cm_val = confusion_matrix(y_pred=val_pred, 
                                      y_true=val_pred_y_batch, 
                                      labels=list(range(NUM_CLASSES)))
            print('')
            print(cm_val)
            # COMPUTE TEST LOSS AND ACCURACY
            print('')
            print('\tEvaluate test performance')
            test_pred, test_pred_y_batch = [], []
            _iter = 1
            #
            for x_batch, y_batch in utils.iterate_minibatches(batchsize=BATCH_SIZE, 
                                                              inputs=inputs_test, 
                                                              targets=targets_test, 
                                                              shuffle=False):
                # LSTM has a static structure...
                if len(y_batch) != BATCH_SIZE: break
                #
                _loss,_acc,_pred = sess.run(fetches=[cross_entropy, accuracy, prediction],
                                            feed_dict={x_pl: x_batch, y_pl: y_batch})
                # append prediction
                test_pred += [np.argmax(_pred[ii]) for ii in range(len(_pred))]
                test_pred_y_batch += [np.argmax(y_batch[ii]) for ii in range(len(y_batch))]
                # append mean
                test_loss.append(_loss)
                test_accuracy.append(_acc)
                print("\t\tminibatch: %d\tLOSS: %f\tACCs: %f" %(_iter,_loss,_acc),end='\r')
                _iter += 1
                # end loop
            # calculate performance
            cm_test = confusion_matrix(y_pred=test_pred, 
                                       y_true=test_pred_y_batch, 
                                       labels=list(range(NUM_CLASSES)))
            print('')
            print(cm_test)
            # CAPTURE STATS FOR CURRENT FOLD
            capture_dict[fold] = {'idx_train': idx_train,
                                  'idx_test': idx_test,
                                  'idx_val': idx_val,
                                  'cm_test': cm_test,
                                  'cm_val': cm_val,
                                  'train_loss': train_loss,
                                  'train_accuracy': train_accuracy,
                                  'test_loss': test_loss,
                                  'test_accuracy': test_accuracy,
                                  'valid_loss': valid_loss,
                                  'valid_accuracy': valid_accuracy}
              
            # SAVE STATS FOR CURRENT FOLD
            np.savez_compressed(MODEL_PATH + "/capture_dict", capture_dict)
            # tf model
            save_path = tf.train.Saver().save(sess, MODEL_PATH + '/fold_' + str(fold) + '.ckpt')
            print("Model saved in file: %s" % save_path)
            # increase fold
            fold += 1
            break
        # end loop and traning    
        print('\n... end training loop')
        print('started at: ' + START_TIME)
        print('ended at:   ' + time.ctime())
        # close session
        sess.close()

    except KeyboardInterrupt:
        pass


