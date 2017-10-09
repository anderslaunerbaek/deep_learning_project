# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 15:27:15 2016

@author: alvmu
"""
# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

import lasagne
import numpy as np
import matplotlib.pyplot as plt
import random

import theano
import theano.tensor as T

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.nonlinearities import softmax

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
import skimage.transform
from skimage import io
import pickle
import imp
utils = imp.load_source('utils', 'utils.py')

#Constants
num_epochs = 50
minibatch_size = 75 #250
lr = 0.00001
num_subjects = 10 #num_subjects_train + num_subjects_val + num_subjects_test
output_nodes = 5
sensors='fpz'
train_features = False
pre_trained_weights = True

# Prepare Theano Symbolic variables for inputs and targets
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')
learning_var = T.scalar('learning')

#VGG model
def build_model():
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224),input_var=input_var)
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=output_nodes, nonlinearity=None) # Original is 1000 units
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)
    l_out = net['prob']
    
    #Set all low layers to be non-trainable
    net['conv1_1'].add_param(net['conv1_1'].W, net['conv1_1'].W.get_value().shape, trainable=train_features)
    net['conv1_1'].add_param(net['conv1_1'].b, net['conv1_1'].b.get_value().shape, trainable=train_features)
    net['conv1_2'].add_param(net['conv1_2'].W, net['conv1_2'].W.get_value().shape, trainable=train_features)
    net['conv1_2'].add_param(net['conv1_2'].b, net['conv1_2'].b.get_value().shape, trainable=train_features)
    net['conv2_1'].add_param(net['conv2_1'].W, net['conv2_1'].W.get_value().shape, trainable=train_features)
    net['conv2_1'].add_param(net['conv2_1'].b, net['conv2_1'].b.get_value().shape, trainable=train_features)
    net['conv2_2'].add_param(net['conv2_2'].W, net['conv2_2'].W.get_value().shape, trainable=train_features)
    net['conv2_2'].add_param(net['conv2_2'].b, net['conv2_2'].b.get_value().shape, trainable=train_features)
    net['conv3_1'].add_param(net['conv3_1'].W, net['conv3_1'].W.get_value().shape, trainable=train_features)
    net['conv3_1'].add_param(net['conv3_1'].b, net['conv3_1'].b.get_value().shape, trainable=train_features)
    net['conv3_2'].add_param(net['conv3_2'].W, net['conv3_2'].W.get_value().shape, trainable=train_features)
    net['conv3_2'].add_param(net['conv3_2'].b, net['conv3_2'].b.get_value().shape, trainable=train_features)
    net['conv3_3'].add_param(net['conv3_3'].W, net['conv3_3'].W.get_value().shape, trainable=train_features)
    net['conv3_3'].add_param(net['conv3_3'].b, net['conv3_3'].b.get_value().shape, trainable=train_features)
    net['conv4_1'].add_param(net['conv4_1'].W, net['conv4_1'].W.get_value().shape, trainable=train_features)
    net['conv4_1'].add_param(net['conv4_1'].b, net['conv4_1'].b.get_value().shape, trainable=train_features)
    net['conv4_2'].add_param(net['conv4_2'].W, net['conv4_2'].W.get_value().shape, trainable=train_features)
    net['conv4_2'].add_param(net['conv4_2'].b, net['conv4_2'].b.get_value().shape, trainable=train_features)
    net['conv4_3'].add_param(net['conv4_3'].W, net['conv4_3'].W.get_value().shape, trainable=train_features)
    net['conv4_3'].add_param(net['conv4_3'].b, net['conv4_3'].b.get_value().shape, trainable=train_features)
    net['conv5_1'].add_param(net['conv5_1'].W, net['conv5_1'].W.get_value().shape, trainable=train_features)
    net['conv5_1'].add_param(net['conv5_1'].b, net['conv5_1'].b.get_value().shape, trainable=train_features)
    net['conv5_2'].add_param(net['conv5_2'].W, net['conv5_2'].W.get_value().shape, trainable=train_features)
    net['conv5_2'].add_param(net['conv5_2'].b, net['conv5_2'].b.get_value().shape, trainable=train_features)
    net['conv5_3'].add_param(net['conv5_3'].W, net['conv5_3'].W.get_value().shape, trainable=train_features)
    net['conv5_3'].add_param(net['conv5_3'].b, net['conv5_3'].b.get_value().shape, trainable=train_features)
#    net['fc6'].add_param(net['fc6'].W, net['fc6'].W.get_value().shape, trainable=False)
#    net['fc6'].add_param(net['fc6'].b, net['fc6'].b.get_value().shape, trainable=False)
#    net['fc7'].add_param(net['fc7'].W, net['fc7'].W.get_value().shape, trainable=False)
#    net['fc7'].add_param(net['fc7'].b, net['fc7'].b.get_value().shape, trainable=False)
    
           
    prediction = lasagne.layers.get_output(l_out)
    loss1 = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss1.sum()
    acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
                  dtype=theano.config.floatX)
    
    # Create update expressions for training:
    params = lasagne.layers.get_all_params(l_out, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=learning_var, beta1=0.9, beta2=0.999)
    
    # Compile a function and returning the corresponding training loss:
    print("Compiling theano functions...")
    # The first two instructions call fast conv3d_fft when running on GPU. Otherwise, conv3D is called
    mode = theano.compile.get_default_mode()
    mode = mode.including('conv3d_fft', 'convgrad3d_fft', 'convtransp3d_fft')
    train_fn = theano.function([input_var, target_var, learning_var], [loss, prediction, acc], updates=updates, mode=mode)
    
    test_prediction = lasagne.layers.get_output(l_out, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.sum()
    test_acc=T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)
    test_fn = theano.function([input_var, target_var], [test_loss, test_prediction, test_acc])
    
    return l_out, train_fn, test_fn

#Load pre-trained parameters on Imagenet
def load_pretrained_parameters(l_out):    

    params = pickle.load(open('../Data/vgg16.pkl'))
    
    #Reset FC6 layer
    fan_in=25088
    fan_out=4096
    a=np.sqrt(12.0/(fan_in+fan_out))
    params['param values'][26]=np.random.uniform(-a,a,(fan_in,fan_out))
    params['param values'][27]=np.zeros((fan_out))

    #Reset FC7 layer 
    fan_in=4096
    fan_out=4096
    a=np.sqrt(12.0/(fan_in+fan_out))
    params['param values'][28]=np.random.uniform(-a,a,(fan_in,fan_out))
    params['param values'][29]=np.zeros((fan_out))   
    
    #Reset FC8 layer
    #We need to remove the weights of last layer, since now we only have 5 output nodes
    #Initialize using Xavier init
    fan_in=4096
    fan_out=output_nodes
    sigma=np.sqrt(2.0/(fan_in+fan_out))
    params['param values'][30]=np.random.normal(0,sigma,(fan_in,fan_out))
    params['param values'][31]=np.zeros((fan_out))
    
    lasagne.layers.set_all_param_values(l_out, map(np.float32, params['param values']))
    CLASSES = params['synset words']
    MEAN_IMAGE = params['mean value']
    
    return l_out, CLASSES, MEAN_IMAGE
    
def random_param_initialisation(l_out):
    #All layers are already randomly initialised. We only need to modify the last one
    
    params = lasagne.layers.get_all_param_values(l_out)
    
    #Reset FC8 layer
    #We need to remove the weights of last layer, since now we only have 1 output
    #Initialize using Xavier init
    fan_in=4096
    fan_out=output_nodes
    sigma=np.sqrt(2.0/(fan_in+fan_out))
    params[30]=np.random.normal(0,sigma,(fan_in,fan_out))
    params[31]=np.zeros((fan_out))
    
    lasagne.layers.set_all_param_values(l_out, map(np.float32, params))
    
    return l_out
    
def load_spectrograms(subject_id, night_id):
    labels = np.loadtxt('../Data/PhysioNet_Img_age_'+sensors+'/sub'+str(subject_id)+'_n'+str(night_id)+'_img_'+sensors+'/labels.txt',dtype='str')
    num_images = np.size(labels)
    
    targets = np.zeros((num_images), dtype='uint8')
    targets[:]=-1    
    targets[labels=='W'] = 0
    targets[labels=='1'] = 1
    targets[labels=='2'] = 2
    targets[labels=='3'] = 3
    targets[labels=='4'] = 4
    targets[labels=='R'] = 5
    
    targets = targets[targets!=-1]
    num_images = np.size(targets)
    
    inputs = np.zeros((num_images,3,224,224),dtype='uint8')

    for idx in range(1,num_images+1):    
        rawim = io.imread('../Data/PhysioNet_Img_age_'+sensors+'/sub'+str(subject_id)+'_n'+str(night_id)+'_img_'+sensors+'/img_'+ np.str(idx) +'.png')
        rawim = rawim[:,:,0:3]
        
        h, w, _ = rawim.shape
        if not (h==224 and w==224):
            rawim = skimage.transform.resize(rawim, (224, 224), preserve_range=True)
        
        # Shuffle axes to c01
        im = np.transpose(rawim,(2,0,1))
        
        im = im[np.newaxis]        
        inputs[idx-1,:,:,:]=im
    
    
    return inputs, targets
    
# Main Function
if __name__ == '__main__':
    
    l_out, train_fn, test_fn = build_model()
    
    if pre_trained_weights:
        print("Using pre-trained weights")
        l_out, CLASSES, MEAN_IMAGE = load_pretrained_parameters(l_out)
    else:
        print("Using randomly initialised weights")
        l_out = random_param_initialisation(l_out)
    
    init_par = lasagne.layers.get_all_param_values(l_out)
    
    #Load all subjects into memory
    subjects_list = []
    for i in range(1,num_subjects+1):
        print("Loading subject %d..." %(i))
        inputs_night1, targets_night1  = load_spectrograms(i,1)
        if(i!=20):
            inputs_night2, targets_night2  = load_spectrograms(i,2)
        else:
            inputs_night2 = np.empty((0,3,224,224),dtype='uint8')
            targets_night2 = np.empty((0,),dtype='uint8')           
        
        current_inputs = np.concatenate((inputs_night1,inputs_night2),axis=0)
        current_targets = np.concatenate((targets_night1, targets_night2),axis=0)
        
        mean_images = np.zeros((5,3,224,224))
        mean_images[0,] = np.mean(current_inputs[current_targets==0,], axis=0)
        mean_images[1,] = np.mean(current_inputs[current_targets==1,], axis=0)
        mean_images[2,] = np.mean(current_inputs[current_targets==2,], axis=0)
        mean_images[3,] = np.mean(current_inputs[np.logical_or(current_targets==3, current_targets==4),], axis=0)
        mean_images[4,] = np.mean(current_inputs[current_targets==5,], axis=0)
        
        subjects_list.append([current_inputs,current_targets, mean_images])

    loo=LeaveOneOut()
    fold=1
    for idx_tmp, idx_test in loo.split(range(num_subjects)):

        print("Fold num %d\tSubject id %d" %(fold, idx_test+1))
        f = open('outputs/sleep5_fold'+str(fold), 'w').close()
        lasagne.layers.set_all_param_values(l_out,init_par)
        random.shuffle(idx_tmp)
        idx_train = idx_tmp[0:15]
        idx_val = idx_tmp[15:19]
                
        num_subjects_train = np.size(idx_train)
        num_subjects_val = np.size(idx_val)
        num_subjects_test = np.size(idx_test)

        train_data = [subjects_list[i] for i in idx_train]
        inputs_train = np.empty((0,3,224,224),dtype='uint8')  
        targets_train = np.empty((0,),dtype='uint8') 
#        mean_train0 = np.empty((0,3,224,224),dtype='uint8')
#        mean_train1 = np.empty((0,3,224,224),dtype='uint8')
#        mean_train2 = np.empty((0,3,224,224),dtype='uint8')
#        mean_train3 = np.empty((0,3,224,224),dtype='uint8')
#        mean_train4 = np.empty((0,3,224,224),dtype='uint8')
        
        for item in train_data:
            inputs_train = np.concatenate((inputs_train,item[0]),axis=0)
            targets_train = np.concatenate((targets_train,item[1]),axis=0)
#            mean_train0 = np.concatenate((mean_train0,item[2][0:1]),axis=0)
#            mean_train1 = np.concatenate((mean_train1,item[2][1:2]),axis=0)
#            mean_train2 = np.concatenate((mean_train2,item[2][2:3]),axis=0)
#            mean_train3 = np.concatenate((mean_train3,item[2][3:4]),axis=0)
#            mean_train4 = np.concatenate((mean_train4,item[2][4:5]),axis=0)
            
#        mean_train0 = np.mean(mean_train0,axis=0,keepdims=True)
#        mean_train1 = np.mean(mean_train1,axis=0,keepdims=True)
#        mean_train2 = np.mean(mean_train2,axis=0,keepdims=True)
#        mean_train3 = np.mean(mean_train3,axis=0,keepdims=True)
#        mean_train4 = np.mean(mean_train4,axis=0,keepdims=True)
        
        val_data = [subjects_list[i] for i in idx_val]
        inputs_val = np.empty((0,3,224,224),dtype='uint8')  
        targets_val = np.empty((0,),dtype='uint8')
#        mean_val0 = np.empty((0,3,224,224),dtype='uint8')
#        mean_val1 = np.empty((0,3,224,224),dtype='uint8')
#        mean_val2 = np.empty((0,3,224,224),dtype='uint8')
#        mean_val3 = np.empty((0,3,224,224),dtype='uint8')
#        mean_val4 = np.empty((0,3,224,224),dtype='uint8')
        for item in val_data:
            inputs_val = np.concatenate((inputs_val,item[0]),axis=0)
            targets_val = np.concatenate((targets_val,item[1]),axis=0)
#            mean_val0 = np.concatenate((mean_val0,item[2][0:1]),axis=0)
#            mean_val1 = np.concatenate((mean_val1,item[2][1:2]),axis=0)
#            mean_val2 = np.concatenate((mean_val2,item[2][2:3]),axis=0)
#            mean_val3 = np.concatenate((mean_val3,item[2][3:4]),axis=0)
#            mean_val4 = np.concatenate((mean_val4,item[2][4:5]),axis=0)
        
#        mean_val0 = np.mean(mean_val0,axis=0,keepdims=True)
#        mean_val1 = np.mean(mean_val1,axis=0,keepdims=True)
#        mean_val2 = np.mean(mean_val2,axis=0,keepdims=True)
#        mean_val3 = np.mean(mean_val3,axis=0,keepdims=True)
#        mean_val4 = np.mean(mean_val4,axis=0,keepdims=True)
            
        test_data = [subjects_list[i] for i in idx_test]
        inputs_test = np.empty((0,3,224,224),dtype='uint8')  
        targets_test = np.empty((0,),dtype='uint8')
#        mean_test0 = np.empty((0,3,224,224),dtype='uint8')
#        mean_test1 = np.empty((0,3,224,224),dtype='uint8')
#        mean_test2 = np.empty((0,3,224,224),dtype='uint8')
#        mean_test3 = np.empty((0,3,224,224),dtype='uint8')
#        mean_test4 = np.empty((0,3,224,224),dtype='uint8')
        
        for item in test_data:
            inputs_test = np.concatenate((inputs_test,item[0]),axis=0)
            targets_test = np.concatenate((targets_test,item[1]),axis=0)  
#            mean_test0 = np.concatenate((mean_test0,item[2][0:1]),axis=0)
#            mean_test1 = np.concatenate((mean_test1,item[2][1:2]),axis=0)
#            mean_test2 = np.concatenate((mean_test2,item[2][2:3]),axis=0)
#            mean_test3 = np.concatenate((mean_test3,item[2][3:4]),axis=0)
#            mean_test4 = np.concatenate((mean_test4,item[2][4:5]),axis=0)
            
        best_loss = float('inf')
        for epoch in range(num_epochs):
            
            ######### Retrain Network ###########   
            print("Retrain network...")
            
            #Select W, N1, N2, N3, REM           
            idx0 = (targets_train==0)
            idx1 = (targets_train==1)
            idx2 = (targets_train==2)            
            idx3 = np.logical_or(targets_train==3, targets_train==4)
            idx4 = (targets_train==5)                   
                        
            num_samples0=np.sum(idx0==True)
            num_samples1=np.sum(idx1==True)
            num_samples2=np.sum(idx2==True)
            num_samples3=np.sum(idx3==True)
            num_samples4=np.sum(idx4==True)
            inputs_tr0 = inputs_train[idx0,]            
            inputs_tr1 = inputs_train[idx1,]
            inputs_tr2 = inputs_train[idx2,]
            inputs_tr3 = inputs_train[idx3,]
            inputs_tr4 = inputs_train[idx4,]
            
            min_samples = np.min((num_samples0, num_samples1, num_samples2, num_samples3, num_samples4))
            
            idx0 = np.random.choice(range(num_samples0),min_samples,replace=False)
            idx1 = np.random.choice(range(num_samples1),min_samples,replace=False)
            idx2 = np.random.choice(range(num_samples2),min_samples,replace=False)
            idx3 = np.random.choice(range(num_samples3),min_samples,replace=False)
            idx4 = np.random.choice(range(num_samples4),min_samples,replace=False)
            
            inputs_tr0 = inputs_tr0[idx0,]
            inputs_tr1 = inputs_tr1[idx1,]
            inputs_tr2 = inputs_tr2[idx2,]
            inputs_tr3 = inputs_tr3[idx3,]
            inputs_tr4 = inputs_tr4[idx4,]
            
            inputs_tr = np.concatenate((inputs_tr0, inputs_tr1, inputs_tr2, inputs_tr3, inputs_tr4))
            targets_tr = np.uint8(np.concatenate((np.zeros((min_samples,),dtype='uint8'),np.ones((min_samples,),dtype='uint8'), 
                                                     np.repeat(2,(min_samples,)),np.repeat(3,(min_samples,)),np.repeat(4,(min_samples,)))))
                                                     
            #Calculate mean and store it as float32
            #mean_train = np.mean(inputs_tr,axis=0,keepdims=True)/255.0
#            mean_train = np.mean(np.concatenate((mean_train0, mean_train1, mean_train2, mean_train3, mean_train4), axis=0), axis=0, keepdims=True)/255.0

            num_batch = 0
            total_acc = 0
            total_loss = 0
            for batch in utils.iterate_minibatches(inputs_tr, targets_tr, minibatch_size, shuffle=True):
                inputs, targets = batch
                
                if np.size(targets)>0:
                
                    #Convert from uint8 to float32
                    inputs = np.float32(inputs)/255.0               
                    #inputs -= mean_train
                    
                    [loss, prediction, acc] = train_fn(inputs, targets, lr) 
                    train_pred=np.argmax(prediction,axis=1)
                    acc0 = np.sum(train_pred[targets==0]==0,dtype='float32')/np.sum(targets==0)
                    acc1 = np.sum(train_pred[targets==1]==1,dtype='float32')/np.sum(targets==1)
                    acc2 = np.sum(train_pred[targets==2]==2,dtype='float32')/np.sum(targets==2)
                    acc3 = np.sum(train_pred[targets==3]==3,dtype='float32')/np.sum(targets==3)
                    acc4 = np.sum(train_pred[targets==4]==4,dtype='float32')/np.sum(targets==4)
                    
                    total_loss += loss
                    total_acc += acc
                    num_batch += 1
                    print("Ep: %d\t L: %f\t ACCs: %f/%d\t %f/%d\t %f/%d\t %f/%d\t %f/%d\t %f/%d" %(epoch, loss, acc,np.size(targets), 
                               acc0, np.sum(targets==0), acc1, np.sum(targets==1), 
                                acc2, np.sum(targets==2), acc3, np.sum(targets==3), acc4, np.sum(targets==4)))
                            
            current_train_loss = total_loss/np.size(targets_tr)
            current_train_acc = total_acc/num_batch
            
            ######### Validate Network ###########            
            print("Validate network...")
            
            #Select only W, N1, N2, N3, REM balanced            
            idx0 = (targets_val==0)
            idx1 = (targets_val==1)
            idx2 = (targets_val==2)            
            idx3 = np.logical_or(targets_val==3, targets_val==4)
            idx4 = (targets_val==5)
            
            num_samples0=np.sum(idx0==True)
            num_samples1=np.sum(idx1==True)
            num_samples2=np.sum(idx2==True)
            num_samples3=np.sum(idx3==True)
            num_samples4=np.sum(idx4==True)
            inputs_v0 = inputs_val[idx0,]            
            inputs_v1 = inputs_val[idx1,]
            inputs_v2 = inputs_val[idx2,]
            inputs_v3 = inputs_val[idx3,]
            inputs_v4 = inputs_val[idx4,]
            
#            min_samples = np.min((num_samples0, num_samples1, num_samples2, num_samples3, num_samples4))
#            
#            idx0 = np.random.choice(range(num_samples0),min_samples,replace=False)
#            idx1 = np.random.choice(range(num_samples1),min_samples,replace=False)
#            idx2 = np.random.choice(range(num_samples2),min_samples,replace=False)
#            idx3 = np.random.choice(range(num_samples3),min_samples,replace=False)
#            idx4 = np.random.choice(range(num_samples4),min_samples,replace=False)
#            
#            inputs_val0 = inputs_val0[idx0,]
#            inputs_val1 = inputs_val1[idx1,]
#            inputs_val2 = inputs_val2[idx2,]
#            inputs_val3 = inputs_val3[idx3,]
#            inputs_val4 = inputs_val4[idx4,]
            
            inputs_v = np.concatenate((inputs_v0, inputs_v1, inputs_v2, inputs_v3, inputs_v4))
            targets_v = np.uint8(np.concatenate((np.zeros((num_samples0,),dtype='uint8'),np.ones((num_samples1,),dtype='uint8'), 
                                                   np.repeat(2,(num_samples2)),np.repeat(3,(num_samples3)),np.repeat(4,(num_samples4)))))
            
            #Calculate mean and store as float
           # mean_val = np.mean(inputs_v,axis=0,keepdims=True)/255.0
#            mean_val = np.mean(np.concatenate((mean_val0, mean_val1, mean_val2, mean_val3, mean_val4), axis=0), axis=0, keepdims=True)/255.0

              
            num_batch = 0
            total_acc = 0
            total_loss = 0
            for batch in utils.iterate_minibatches(inputs_v, targets_v, minibatch_size, shuffle=True):
                inputs, targets = batch
                
                if np.size(targets)>0:
                
                    #Convert from uint8 to float32
                    inputs = np.float32(inputs)/255.0                
                   # inputs -= mean_val
                    
                    [loss, prediction, acc] = test_fn(inputs, targets)
                    test_pred=np.argmax(prediction,axis=1)
                    acc0 = np.sum(test_pred[targets==0]==0,dtype='float32')/np.sum(targets==0)
                    acc1 = np.sum(test_pred[targets==1]==1,dtype='float32')/np.sum(targets==1)
                    acc2 = np.sum(test_pred[targets==2]==2,dtype='float32')/np.sum(targets==2)
                    acc3 = np.sum(test_pred[targets==3]==3,dtype='float32')/np.sum(targets==3)
                    acc4 = np.sum(test_pred[targets==4]==4,dtype='float32')/np.sum(targets==4)
                    
                    total_loss += loss
                    total_acc += acc
                    num_batch += 1
                    print("Val\t L: %f\t ACCs: %f/%d\t %f/%d\t %f/%d\t %f/%d\t %f/%d\t %f/%d" %(loss, acc, np.size(targets), acc0, np.sum(targets==0), acc1,np.sum(targets==1), 
                                acc2, np.sum(targets==2), acc3, np.sum(targets==3),acc4, np.sum(targets==4)))
                            
            current_val_loss = total_loss/np.size(targets_v)
            current_val_acc = total_acc/num_batch
            
            f = open('outputs/sleep5_fold'+str(fold), 'a')
            f.write('\n'+str(current_train_loss)+'\t'+str(current_train_acc)+'\t'+str(current_val_loss)+'\t'+str(current_val_acc))
            f.close()
                            
            ######### Test Network ###########
            if(current_val_loss < best_loss):
                best_loss=current_val_loss
                print("Test network...")
                    
                #Select only W, N1, N2, N3, REM balanced            
                idx0 = (targets_test==0)
                idx1 = (targets_test==1)
                idx2 = (targets_test==2)            
                idx3 = np.logical_or(targets_test==3, targets_test==4)
                idx4 = (targets_test==5)            
                
                num_samples0=np.sum(idx0==True)
                num_samples1=np.sum(idx1==True)
                num_samples2=np.sum(idx2==True)
                num_samples3=np.sum(idx3==True)
                num_samples4=np.sum(idx4==True)
                inputs_te0 = inputs_test[idx0,]            
                inputs_te1 = inputs_test[idx1,]
                inputs_te2 = inputs_test[idx2,]
                inputs_te3 = inputs_test[idx3,]
                inputs_te4 = inputs_test[idx4,]
                
#                min_samples = np.min((num_samples0, num_samples1, num_samples2, num_samples3, num_samples4))
#                
#                idx0 = np.random.choice(range(num_samples0),min_samples,replace=False)
#                idx1 = np.random.choice(range(num_samples1),min_samples,replace=False)
#                idx2 = np.random.choice(range(num_samples2),min_samples,replace=False)
#                idx3 = np.random.choice(range(num_samples3),min_samples,replace=False)
#                idx4 = np.random.choice(range(num_samples4),min_samples,replace=False)
#                
#                inputs_test0 = inputs_test0[idx0,]
#                inputs_test1 = inputs_test1[idx1,]
#                inputs_test2 = inputs_test2[idx2,]
#                inputs_test3 = inputs_test3[idx3,]
#                inputs_test4 = inputs_test4[idx4,]
                
                inputs_te = np.concatenate((inputs_te0, inputs_te1, inputs_te2, inputs_te3, inputs_te4))
                targets_te = np.uint8(np.concatenate((np.zeros((num_samples0,),dtype='uint8'),np.ones((num_samples1,),dtype='uint8'), 
                                                       np.repeat(2,(num_samples2)),np.repeat(3,(num_samples3)),np.repeat(4,(num_samples4)))))
                
                #Calculate mean and store it as float32
                #mean_test = np.mean(inputs_te,axis=0,keepdims=True)/255.0
#                mean_test = np.mean(np.concatenate((mean_test0, mean_test1, mean_test2, mean_test3, mean_test4), axis=0), axis=0, keepdims=True)/255.0

                
                num_batch = 0
                total_acc = 0
                total_loss = 0
                total_prediction_test = np.empty((0,),dtype='uint8')
                used_targets = np.empty((0,),dtype='uint8')
                for batch in utils.iterate_minibatches(inputs_te, targets_te, minibatch_size, shuffle=True):
                    inputs, targets = batch
                    
                    if np.size(targets)>0:
                    
                        #Convert from uint8 to float32
                        inputs = np.float32(inputs)/255.0
                       # inputs -= mean_test
                        
                        [loss, prediction, acc] = test_fn(inputs, targets)
                        test_pred=np.argmax(prediction,axis=1)
                        total_prediction_test = np.concatenate((total_prediction_test,test_pred),axis=0)
                        used_targets = np.concatenate((used_targets,targets),axis=0)
                        acc0 = np.sum(test_pred[targets==0]==0,dtype='float32')/np.sum(targets==0)
                        acc1 = np.sum(test_pred[targets==1]==1,dtype='float32')/np.sum(targets==1)
                        acc2 = np.sum(test_pred[targets==2]==2,dtype='float32')/np.sum(targets==2)
                        acc3 = np.sum(test_pred[targets==3]==3,dtype='float32')/np.sum(targets==3)
                        acc4 = np.sum(test_pred[targets==4]==4,dtype='float32')/np.sum(targets==4)
                        
                        total_loss += loss
                        total_acc += acc
                        num_batch += 1
                        print("Test\t L: %f\t ACCs: %f/%d\t %f/%d\t %f/%d\t %f/%d\t %f/%d\t %f/%d" %(loss, acc, np.size(targets), acc0, np.sum(targets==0), acc1,np.sum(targets==1), 
                                    acc2, np.sum(targets==2), acc3, np.sum(targets==3),acc4, np.sum(targets==4)))
                
                current_test_loss = total_loss/np.size(targets_te)
                current_test_acc = total_acc/num_batch                
                
                f = open('outputs/sleep5_fold'+str(fold), 'a')
                f.write('\t'+str(current_test_loss)+'\t'+str(current_test_acc))
                f.close()
                
                cm = confusion_matrix(used_targets, total_prediction_test, labels=None, sample_weight=None)
                np.savetxt('outputs/confusion_matrix_'+str(fold)+'.txt',cm,fmt='%d', delimiter='\t')
                
                best_pars = lasagne.layers.get_all_param_values(l_out)
                
        #Save best params. This instruction takes some time. That is why we do it only when the fold is finished.
        np.savez_compressed('outputs/params_'+str(fold),best_pars[26:32])
        fold+=1    
    print("Done!")