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

import skimage.transform
from skimage import io
import pickle
import imp
utils = imp.load_source('utils', 'utils.py')

#Constants
minibatch_size = 50
num_subjects = 20 #num_subjects_train + num_subjects_val + num_subjects_test
subject_to_evaluate = 7
output_nodes = 5
sensors='fpz'
train_features=False

# Prepare Theano Symbolic variables for inputs and targets
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

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
    net['prob'] = NonlinearityLayer(net['fc8'], softmax) #We change softmax by sigmoid
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
  
    
    # Compile a function and returning the corresponding training loss:
    print("Compiling theano functions...")
    
    prediction = lasagne.layers.get_output(l_out,deterministic=True)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()    

    # Gradient of realignment error wrt inputs
    grad_output_wrt_input = T.grad(loss, wrt=[input_var])
    sensitivity_fn = theano.function([input_var, target_var], [grad_output_wrt_input[0],prediction])
    
    return l_out, sensitivity_fn

#Load pre-trained parameters on Imagenet
def load_pretrained_parameters(l_out):    

    params = pickle.load(open('../Data/vgg16.pkl'))
    params_specific = np.load('outputs/params_'+str(subject_to_evaluate)+'.npz')['arr_0']
    
#    #Set FC6 layer
#    params['param values'][26]=params_specific[0]
#    params['param values'][27]=params_specific[1]
#
#    #Set FC7 layer 
#    params['param values'][28]=params_specific[2]
#    params['param values'][29]=params_specific[3]
#    
#    #Set FC8 layer
#    params['param values'][30]=params_specific[4]
#    params['param values'][31]=params_specific[5]
    
    params['param_values'] = params_specific
    
    #lasagne.layers.set_all_param_values(l_out, map(np.float32, params['param values']))
    lasagne.layers.set_all_param_values(l_out, params_specific)
    CLASSES = params['synset words']
    MEAN_IMAGE = params['mean value']
    
    return l_out, CLASSES, MEAN_IMAGE
    
def load_spectrograms(subject_id, night_id):
    labels = np.loadtxt('../Data/PhysioNet_Img_age_'+sensors+'/sub'+str(subject_id)+'_n'+str(night_id)+'_img_'+sensors+'/labels.txt',dtype='str')
    num_images = np.size(labels)
    
    targets = np.zeros((num_images), dtype='int32')
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
    
    mean_image = np.mean(inputs, axis=0, keepdims=True)
    return inputs, targets, mean_image
    
    
# Main Function
if __name__ == '__main__':
    
    l_out, sensitivity_fn = build_model()
    l_out, CLASSES, MEAN_IMAGE = load_pretrained_parameters(l_out)  
    
    #init_par = lasagne.layers.get_all_param_values(l_out)
    
    #Load all subjects in memory
    subjects_list = []
    for i in range(1,num_subjects+1):
        print("Loading subject %d..." %(i))
        inputs_night1, targets_night1, _  = load_spectrograms(i,1)
        if(i!=20):
            inputs_night2, targets_night2, _  = load_spectrograms(i,2)
        else:
            inputs_night2 = np.empty((0,3,224,224),dtype='uint8')
            targets_night2 = np.empty((0,),dtype='int32')
        
        current_inputs = np.concatenate((inputs_night1,inputs_night2),axis=0)
        current_targets = np.concatenate((targets_night1, targets_night2),axis=0)        
        
        subjects_list.append([current_inputs,current_targets])
   
    idx_train = range(20)
    idx_train.remove(subject_to_evaluate - 1)
    idx_test = subject_to_evaluate - 1
    train_data = [subjects_list[i] for i in idx_train]
    inputs_train = np.empty((0,3,224,224),dtype='uint8')  
    targets_train = np.empty((0,),dtype='int32')       
    for item in train_data:
        inputs_train = np.concatenate((inputs_train,item[0]),axis=0)
        targets_train = np.concatenate((targets_train,item[1]),axis=0)
        
    mean_train = np.mean(inputs_train,axis=0,keepdims=True)/255.0
            
    current_subject = subjects_list[idx_test]            
    inputs_test = current_subject[0]
    targets_test = current_subject[1]
       
                        
    ######### Test Network ###########     
    print("Calculate sensitivity maps...")
        
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
    
    
    inputs_te = np.concatenate((inputs_te0, inputs_te1, inputs_te2, inputs_te3, inputs_te4))
    targets_te = np.int32(np.concatenate((np.zeros((num_samples0,),dtype='int32'),np.ones((num_samples1,),dtype='int32'), 
                                           np.repeat(2,(num_samples2)),np.repeat(3,(num_samples3)),np.repeat(4,(num_samples4)))))
    
    #Calculate mean and store it as float32
    mean_test = np.mean(inputs_te,axis=0,keepdims=True)/255.0


    ######## Single sensitivity map
#    inputs = np.float32(inputs_te)/255.0
#    inputs -= mean_train
#    targets = targets_te
#    [grad] = sensitivity_fn(inputs[20:21], targets[20:21])
#    sm = np.mean(np.abs(grad), axis=0) 
#    sm_min=np.min(sm)
#    sm_max=np.max(sm)
#    sensitivity_map = (sm-sm_min)/(sm_max-sm_min)  


    ######### Accumulated sensitivity map
    inputs_te = inputs_te[targets_te==0,]
    targets_te = targets_te[targets_te==0,]
     
    grad_accum = np.empty((0,3,224,224))
    pred_accum = np.empty((0,5))
    num_batch = 0
    for batch in utils.iterate_minibatches(inputs_te, targets_te, minibatch_size, shuffle=False):
        inputs, targets = batch
        print("Batch num %d" %(num_batch))
        
        if np.size(targets)>0: 
        
            #Convert from uint8 to float32
            inputs = np.float32(inputs)/255.0
           # inputs -= mean_train
    
            #Sensitivity map
            [grad,pred] = sensitivity_fn(inputs, targets)
            grad_accum=np.concatenate((grad_accum,grad),axis=0) 
            pred_accum=np.concatenate((pred_accum,pred),axis=0)

        num_batch += 1              
           
    sm = np.mean(np.abs(grad_accum), axis=0)
    sm=np.sum(sm,axis=0)
    sm_min=np.min(sm)
    sm_max=np.max(sm)
    sensitivity_map = (sm-sm_min)/(sm_max-sm_min) 
    print("Done!")
    
    #Plot per-class sensitvity map
    f = plt.figure()
    extent_v = [-60, 90, 0, 30]
    plt.imshow(sensitivity_map, interpolation=None, extent=extent_v)
    plt.axes().set_aspect(abs((extent_v[1]-extent_v[0])/(extent_v[3]-extent_v[2])))
    plt.xlabel('time (s)')
    plt.ylabel('frequency (Hz)')
    f.savefig("sens_map_R_sub7.pdf", bbox_inches='tight')
    
    #Plot inspected epoch spectrogram
    insp_epoch = 20
    
    f = plt.figure()
    extent_v = [-60, 90, 0, 30]
    plt.imshow(np.transpose(inputs_te[insp_epoch],(1,2,0)), interpolation=None, extent=extent_v)
    plt.axes().set_aspect(abs((extent_v[1]-extent_v[0])/(extent_v[3]-extent_v[2])))
    plt.xlabel('time (s)')
    plt.ylabel('frequency (Hz)')
    f.savefig("good_spec_R_sub7.pdf", bbox_inches='tight')
    
    #Plot corresponding sensitivity map
    ind_grad, _ = sensitivity_fn(inputs_te[insp_epoch:insp_epoch+1,], targets_te[insp_epoch:insp_epoch+1,])
    ind_sm = np.abs(ind_grad)
    ind_sm = np.sum(ind_sm[0],axis=0)
    ind_sm_min=np.min(ind_sm)
    ind_sm_max=np.max(ind_sm)
    ind_sensitivity_map = (ind_sm-ind_sm_min)/(ind_sm_max-ind_sm_min)
    
    f = plt.figure()
    extent_v = [-60, 90, 0, 30]
    plt.imshow(ind_sensitivity_map, interpolation=None, extent=extent_v)
    plt.axes().set_aspect(abs((extent_v[1]-extent_v[0])/(extent_v[3]-extent_v[2])))
    plt.xlabel('time (s)')
    plt.ylabel('frequency (Hz)')
    f.savefig("good_sm_R_sub7.pdf", bbox_inches='tight')
    
   
    
    
    
    
    
    
    

    
