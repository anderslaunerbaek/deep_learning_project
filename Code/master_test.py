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
NUM_SUBJECTS = 2
NUM_CLASSES = 6
VAL_TRAIN_ID = NUM_SUBJECTS - 1

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

## Building the model ----
# hyper parameters
HEIGTH, WIDTH, NCHANNELS = IMAGE_SHAPE[1], IMAGE_SHAPE[2], IMAGE_SHAPE[3]
L_RATE = 10e-5
L_RATE_MO_1 = 0.9
L_RATE_MO_2 = 0.999
EPS = 1e-8
KEEP_PROB = 0.5
# Training Loop
MAX_EPOCHS = 5 # 50
BATCH_SIZE = 250 #

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction = 0.95

# https://www.cs.toronto.edu/~frossard/vgg16/vgg16.py
# Load the weights into memory
weights_dict = np.load(data_dir + '/' + 'vgg16_weights.npz', encoding='bytes')

def tf_conv2d(inputs, name):
    with tf.name_scope(name) as scope:
        weights = tf.get_variable(shape=weights_dict[name + '_W'].shape, 
                                  initializer=tf.constant_initializer(weights_dict[name + '_W']),
                                  name=scope + 'weights', 
                                  trainable=False)
        conv = tf.nn.conv2d(inputs, weights, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(shape=weights_dict[name + '_b'].shape,
                                 initializer=tf.constant_initializer(weights_dict[name + '_b']), 
                                 trainable=False, name=scope + 'biases')
        return(tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope))
        
def tf_max_pooling2d(inputs, name, kh = 2, kw = 2, dh = 2, dw = 2):
    with tf.name_scope(name) as scope:
        return(tf.nn.max_pool(inputs,
                              ksize=[1, kh, kw, 1],
                              strides=[1, dh, dw, 1],
                              padding='VALID',
                              name=scope))        

def tf_fully_con(inputs, name, n_out=4096, train_able = True):
    n_in = n_in = inputs.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        if train_able:
            weights = tf.get_variable(shape=[n_in, n_out],
                                      dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      name=scope + 'weights', 
                                      trainable=True)

            biases = tf.get_variable(shape=n_out,
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0),
                                     trainable=True, 
                                     name=scope + 'biases')
        else:
            weights = tf.get_variable(shape=[n_in, n_out],
                                      dtype=tf.float32,
                                      initializer=tf.constant_initializer(weights_dict[name + '_W']), 
                                      name=scope + 'weights', 
                                      trainable=False)

            biases = tf.get_variable(shape=n_out,
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(weights_dict[name + '_b']), 
                                     trainable=False, 
                                     name=scope + 'biases')
        
        #
        return(tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, weights), biases)))
        
# https://github.com/huyng/tensorflow-vgg/blob/master/layers.py
# init model
tf.reset_default_graph()
# init placeholders
x_pl = tf.placeholder(tf.float32, [None, HEIGTH, WIDTH, NCHANNELS], name='input_placeholder')
y_pl = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='target_placeholder')
print('Trace of the tensors shape as it is propagated through the network.')
print('Layer name \t Output size')
print('--------------------------------------------')
with tf.variable_scope('VVG16_layer'):
    # subtract image mean
    mu = tf.constant(np.array([115.79640507,127.70359263,119.96839583], dtype=np.float32), 
                     name="rgb_mean")
    net = tf.subtract(x_pl, mu, name="input_mean_centered")
    
    # level one
    net = tf_conv2d(inputs=net, name='conv1_1')
    print('conv1_1 \t', net.get_shape())
    net = tf_conv2d(inputs=net, name='conv1_2')
    print('conv1_2 \t', net.get_shape())
    net = tf_max_pooling2d(inputs=net, name='pool1')
    print('pool1 \t\t', net.get_shape())
    print('--------------------------------------------')
    
    # level two
    net = tf_conv2d(inputs=net, name='conv2_1')
    print('conv2_1 \t', net.get_shape())
    net = tf_conv2d(inputs=net, name='conv2_2')
    print('conv2_2 \t', net.get_shape())
    net = tf_max_pooling2d(inputs=net, name='pool2')
    print('pool2 \t\t', net.get_shape())
    print('--------------------------------------------')
    
    # level three
    net = tf_conv2d(inputs=net, name='conv3_1')
    print('conv3_1 \t', net.get_shape())
    net = tf_conv2d(inputs=net, name='conv3_2')
    print('conv3_2 \t', net.get_shape())
    net = tf_conv2d(inputs=net, name='conv3_3')
    print('conv3_3 \t', net.get_shape())
    net = tf_max_pooling2d(inputs=net, name='pool_3')
    print('pool3 \t\t', net.get_shape())
    print('--------------------------------------------')
    
    # level four
    net = tf_conv2d(inputs=net, name='conv4_1')
    print('conv4_1 \t', net.get_shape())
    net = tf_conv2d(inputs=net, name='conv4_2')
    print('conv4_2 \t', net.get_shape())
    net = tf_conv2d(inputs=net, name='conv4_3')
    print('conv4_3 \t', net.get_shape())
    net = tf_max_pooling2d(inputs=net, name='pool_4')
    print('pool4 \t\t', net.get_shape())
    print('--------------------------------------------')

    # level five
    net = tf_conv2d(inputs=net, name='conv5_1')
    print('conv5_1 \t', net.get_shape())
    net = tf_conv2d(inputs=net, name='conv5_2')
    print('conv5_2 \t', net.get_shape())
    net = tf_conv2d(inputs=net, name='conv5_3')
    print('conv5_3 \t', net.get_shape())
    net = tf_max_pooling2d(inputs=net, name='pool_5')
    print('pool5 \t\t', net.get_shape())
    print('--------------------------------------------')
    
    
    # flatten
    flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
    net = tf.reshape(net, [-1, flattened_shape], name="flatten")
    print('flatten \t', net.get_shape())
    # level six
    net = tf_fully_con(inputs=net, name='fc6', n_out=4096)
    print('fc6 \t\t', net.get_shape())
    net = tf.layers.dropout(inputs=net, name='fc6_dropout', rate=KEEP_PROB)

    # level seven
    net = tf_fully_con(inputs=net, name='fc7', n_out=4096)
    print('fc7 \t\t', net.get_shape())
    net = tf.layers.dropout(inputs=net, name='fc7_dropout', rate=KEEP_PROB)

    # level eigth
    logits = tf_fully_con(inputs=net, name='fc8', n_out=NUM_CLASSES)
    print('fc8 \t\t', logits.get_shape()) 
    print('--------------------------------------------')
        
# print trainable 
no_train_able = []
for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='VVG16_layer'):
    no_train_able.append(np.prod(i.shape[:]).value)   # i.name if you want just a name   
print('Model consits of ', np.sum(no_train_able), 'trainable parameters.')


## OPTIMISATION ----

with tf.variable_scope('performance'):
    probs = tf.nn.softmax(logits)
    prediction = tf.one_hot(tf.argmax(probs, axis=1), depth=NUM_CLASSES)
    prediction_bool = tf.equal(tf.argmax(probs, axis=1), tf.argmax(y_pl, axis=1))
    accuracy = tf.reduce_mean(tf.cast(prediction_bool, tf.float32))
    
with tf.variable_scope('loss_function'):
    # computing cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=y_pl,
                                                            name='cross_entropy')
    loss = tf.reduce_mean(cross_entropy)
    # defining our optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=L_RATE,  
                                       beta1=L_RATE_MO_1, 
                                       beta2=L_RATE_MO_2, 
                                       epsilon = EPS)
    # applying the gradients
    train_model = optimizer.minimize(loss)


## Test flow for model ----
# # Launch TensorBoard, and visualize the TF graph
# with tf.Session() as sess:
    # writer = tf.summary.FileWriter(logs_path, sess.graph)
    # close session
    # sess.close()
# run in terminal
# """
# python -m webbrowser "http://localhost:6006/";
# tensorboard --logdir='./logs'
# """

# flow test
if False:
    # Test the forward pass    
    x_batch = subjects_list[0][0][0:40]
    y_batch = subjects_list[0][1][0:40]

    sess = tf.Session(config=config)
    #tf.train.start_queue_runners(sess=sess_test)
    with sess.as_default():
        #
        sess.run(tf.global_variables_initializer())
        #
        tmp_net = sess.run(fetches=net, 
                       feed_dict={x_pl: x_batch,
                                  y_pl: y_batch})

        tmp_pred = sess.run(fetches=prediction, 
                   feed_dict={x_pl: x_batch})

        tmp_pred_cor = sess.run(fetches=prediction_bool, 
                   feed_dict={x_pl: x_batch,
                             y_pl: y_batch})

        tmp_accuracy = sess.run(fetches=accuracy, 
                   feed_dict={x_pl: x_batch,
                             y_pl: y_batch})

        tmp_cross_entropy = sess.run(fetches=cross_entropy, 
                   feed_dict={x_pl: x_batch,
                             y_pl: y_batch})

        tmp_loss = sess.run(fetches=loss, 
                            feed_dict={x_pl: x_batch,
                                      y_pl: y_batch})

        #tmp_grad_output_wrt_input = sess.run(fetches=grad_output_wrt_input, 
        #                    feed_dict={x_pl: x_batch, y_pl: y_batch})

        _loss,_acc,_pred = sess.run(fetches=[loss, accuracy, prediction],
                            feed_dict={x_pl: x_batch, y_pl: y_batch})
        
        #u_s.cal_sen_map(grad_accum=x_batch, IMAGE_SHAPE=IMAGE_SHAPE, sen_map_class='2')
        #u_s.cal_sen_map(grad_accum=tmp_grad_output_wrt_input, IMAGE_SHAPE=IMAGE_SHAPE, sen_map_class='2')
        #x_batch = subjects_list[0][0][0:100]
        #y_batch = subjects_list[0][1][0:100]
        #print(time.ctime())
        #_,tm2,tm3 = sess.run(fetches=[train_model, loss, accuracy],
        #             feed_dict={x_pl: x_batch, y_pl: y_batch})
        #print(time.ctime())
        #x_batch = subjects_list[0][0][0:1]
        #y_batch = subjects_list[0][1][0:1]
        #tmp_grad_output_wrt_input = sess.run(fetches=grad_output_wrt_input, 
        #                    feed_dict={x_pl: x_batch, y_pl: y_batch})
        #u_s.cal_sen_map(grad_accum=tmp_grad_output_wrt_input, IMAGE_SHAPE=IMAGE_SHAPE, sen_map_class='2')
        #u_s.save_weights(graph= tf.get_default_graph(), fpath=data_dir + '/weigths.npz')
        # close session
        sess.close()

    #assert y_pred.shape == np.zeros((len(x_batch),NUM_CLASSES)).shape, "ERROR the output shape is not as expected!" \
    #        + " Output shape should be " + str(l_out.shape) + ' but was ' + str(y_pred.shape)

    print('Forward pass successful!')


## Train the model ----
capture_dict = {}
sess = tf.Session(config=config)
with sess.as_default():
    print('Begin training loop... \n')
    
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
        
        # initlize variables    
        sess.run(tf.global_variables_initializer())
        
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
                                                          shuffle=False):
            #
            _,_loss,_acc = sess.run(fetches=[train_model, loss, accuracy],
                                     feed_dict={x_pl: x_batch, y_pl: y_batch})
            
            # append to mini batch
            _train_loss.append(_loss)
            _train_accuracy.append(_acc)                    
            #
            print("\t\tminibatch: %d\tL: %f\tACCs: %f" %(_iter,
                                                    np.nanmean(_train_loss),
                                                    np.nanmean(_train_accuracy)),end='\r')
            _iter += 1
            break
            # end loo
            # append mean loss and accuracy
            train_loss.append(np.nanmean(_train_loss))
            train_accuracy.append(np.nanmean(_train_accuracy))
            # end loop
    
    # COMPUTE VALIDATION LOSS AND ACCURACY
    print('\tEvaluate validation performance')
    val_pred, val_pred_y_batch = [], []
    _iter = 1
    #
    for x_batch, y_batch in utils.iterate_minibatches(batchsize=BATCH_SIZE, 
                                                      inputs=inputs_val, 
                                                      targets=targets_val, 
                                                      shuffle=True):
        _loss,_acc,_pred = sess.run(fetches=[loss, accuracy, prediction],
                                    feed_dict={x_pl: x_batch, y_pl: y_batch})
        # append prediction
        val_pred += [np.argmax(_pred[ii]) for ii in range(len(_pred))]
        val_pred_y_batch += [np.argmax(y_batch[ii]) for ii in range(len(y_batch))]
        # append mean
        valid_loss.append(_loss)
        valid_accuracy.append(_acc)
        print("\t\tminibatch: %d\tL: %f\tACCs: %f" %(_iter,
                                                np.nanmean(valid_loss),
                                                np.nanmean(valid_accuracy)),end='\r')
        _iter += 1
        break
    #
    print(confusion_matrix(y_pred=val_pred, 
                          y_true=val_pred_y_batch, 
                          labels=list(range(NUM_CLASSES))))

    print("end")
    
        
