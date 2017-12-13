# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 15:27:15 2016

@author: Albert Vilamala
@email: alvmu@dtu.dk
@affiliation: Technical University of Denmark
@url: http://people.compute.dtu.dk/alvmu
"""
# This code was used in the paper:
# Albert Vilamala, Kristoffer H. Madsen, Lars K. Hansen
# "Deep Convolutional Neural Networks for Interpretable Analysis of EEG Sleep Stage Scoring"
# which can be downloaded from: https://arxiv.org/abs/1710.00633

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
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.nonlinearities import softmax

import skimage.transform
from skimage import io

#Constants
minibatch_size = 50
num_subjects = 20 #Total number of subjects
subject_to_evaluate = 7 #Subject over which sensitivity maps are calcualted
output_nodes = 5
sensors='fpz'
train_features=False
class_sm = 0 #Class-specific Sensitivity Map. 0: awake, 1: NonRem1, 2:NonRem2, 3:NonRem3, 4:REM
insp_epoch = 20 #Individual spectrogram Sensitivity Map

# Theano Symbolic variables
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

#VGG-16 model
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
        net['fc7_dropout'], num_units=output_nodes, W=lasagne.init.GlorotNormal(), nonlinearity=None) 
    net['prob'] = NonlinearityLayer(net['fc8'], softmax) 
    l_out = net['prob']
    
    #Set whether low layers are trainable or non-trainable
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
    
    # Define Loss function and accuracy measure for training
    prediction = lasagne.layers.get_output(l_out,deterministic=True)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()    

    # Define function that computes the gradient of output loss wrt inputs
    grad_output_wrt_input = T.grad(loss, wrt=[input_var])
    sensitivity_fn = theano.function([input_var, target_var], [grad_output_wrt_input[0],prediction])
    
    return l_out, sensitivity_fn

#Load pre-trained weights
def load_pretrained_parameters(l_out):    

    #Load weights from trained VGG on EEG spectrogram
    params_specific = np.load('outputs/params_'+str(subject_to_evaluate)+'.npz')['arr_0']

    lasagne.layers.set_all_param_values(l_out, params_specific)
    
    return l_out

#Load spectrogram images
def load_spectrograms(subject_id, night_id):
    
    #Load hypnogram labels
    labels = np.loadtxt('PhysioNet_Sleep_EEG/sub'+str(subject_id)+'_n'+str(night_id)+'_img_'+sensors+'/labels.txt',dtype='str')
    num_images = np.size(labels)
    
    #Stages are: W:awake, 1:Non-Rem1, 2:Non-Rem2, 3:Non-Rem3, 4:Non-Rem4, R:Rem
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
    
    #Load spectrogram images 3 RGB channels of size 224 x 224
    inputs = np.zeros((num_images,3,224,224),dtype='uint8')

    for idx in range(1,num_images+1):    
        rawim = io.imread('PhysioNet_Sleep_EEG/sub'+str(subject_id)+'_n'+str(night_id)+'_img_'+sensors+'/img_'+ np.str(idx) +'.png')
        rawim = rawim[:,:,0:3]
        
        h, w, _ = rawim.shape
        if not (h==224 and w==224):
            rawim = skimage.transform.resize(rawim, (224, 224), preserve_range=True)
        
        # Shuffle axes to match the way pretrained weights were calculated
        im = np.transpose(rawim,(2,0,1))
   
        im = im[np.newaxis]        
        inputs[idx-1,:,:,:]=im
    
    return inputs, targets

#Batch iterator
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    last_idx = batchsize*(len(inputs)/batchsize)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, last_idx+batchsize, batchsize):
        if shuffle:
            if start_idx!=last_idx:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = indices[start_idx:]
        else:
            if start_idx!=last_idx:
                excerpt = slice(start_idx, start_idx + batchsize)
            else:
                excerpt = slice(start_idx, len(inputs))
        yield inputs[excerpt], targets[excerpt]
    
    
# Main Function
if __name__ == '__main__':
    
    #Build Neural Network and sensitivity functions    
    l_out, sensitivity_fn = build_model()
    
    #Load NN weights 
    l_out = load_pretrained_parameters(l_out)  
    
    #Load all subjects in memory
    subjects_list = []
    for i in range(1,num_subjects+1):
        print("Loading subject %d..." %(i))
        inputs_night1, targets_night1  = load_spectrograms(i,1)
        if(i!=20):
            inputs_night2, targets_night2  = load_spectrograms(i,2)
        else:
            inputs_night2 = np.empty((0,3,224,224),dtype='uint8')
            targets_night2 = np.empty((0,),dtype='int32')
        
        current_inputs = np.concatenate((inputs_night1,inputs_night2),axis=0)
        current_targets = np.concatenate((targets_night1, targets_night2),axis=0)        
        
        subjects_list.append([current_inputs,current_targets])
   
    #Select train and test subjects    
    idx_train = range(20)
    idx_train.remove(subject_to_evaluate - 1)
    idx_test = subject_to_evaluate - 1
    train_data = [subjects_list[i] for i in idx_train]
    inputs_train = np.empty((0,3,224,224),dtype='uint8')  
    targets_train = np.empty((0,),dtype='int32')       
    for item in train_data:
        inputs_train = np.concatenate((inputs_train,item[0]),axis=0)
        targets_train = np.concatenate((targets_train,item[1]),axis=0)
        
    current_subject = subjects_list[idx_test]            
    inputs_test = current_subject[0]
    targets_test = current_subject[1]
       
                        
    ######### Test Network ###########     
    print("Calculate sensitivity maps...")
        
    #Select only W, N1, N2, N3, REM balanced            
    idx0 = (targets_test==0)
    idx1 = (targets_test==1)
    idx2 = (targets_test==2)            
    idx3 = np.logical_or(targets_test==3, targets_test==4) # NonRem3 and NonRem4 are unified to N3 according to AASM
    idx4 = (targets_test==5)            
    
    #Split inputs according to class labels
    inputs_te0 = inputs_test[idx0,]            
    inputs_te1 = inputs_test[idx1,]
    inputs_te2 = inputs_test[idx2,]
    inputs_te3 = inputs_test[idx3,]
    inputs_te4 = inputs_test[idx4,]
    
    #Create appropriate inputs and targets
    num_samples0=np.sum(idx0==True)
    num_samples1=np.sum(idx1==True)
    num_samples2=np.sum(idx2==True)
    num_samples3=np.sum(idx3==True)
    num_samples4=np.sum(idx4==True)   
    inputs_te = np.concatenate((inputs_te0, inputs_te1, inputs_te2, inputs_te3, inputs_te4))
    targets_te = np.int32(np.concatenate((np.zeros((num_samples0,),dtype='int32'),np.ones((num_samples1,),dtype='int32'), 
                                           np.repeat(2,(num_samples2)),np.repeat(3,(num_samples3)),np.repeat(4,(num_samples4)))))
    
    ######### Accumulated sensitivity map over whole class
    inputs_te = inputs_te[targets_te==class_sm,]
    targets_te = targets_te[targets_te==class_sm,]
     
    grad_accum = np.empty((0,3,224,224))
    pred_accum = np.empty((0,5))
    num_batch = 0
    for batch in iterate_minibatches(inputs_te, targets_te, minibatch_size, shuffle=False):
        inputs, targets = batch
        print("Batch num %d" %(num_batch))
        
        if np.size(targets)>0: 
        
            #Convert from uint8 to float32
            inputs = np.float32(inputs)/255.0
    
            #Calculate Gradients wrt inputs
            [grad,pred] = sensitivity_fn(inputs, targets)
            grad_accum=np.concatenate((grad_accum,grad),axis=0) 
            pred_accum=np.concatenate((pred_accum,pred),axis=0)

        num_batch += 1              
    
    #Calcualte Sensitivity maps       
    sm = np.mean(np.abs(grad_accum), axis=0)
    sm=np.sum(sm,axis=0)
    
    #Scale between 0 and 1
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
    f.savefig('sens_map_class_'+class_sm+'_sub'+subject_to_evaluate+'.pdf', bbox_inches='tight')
      
    ######## Single sensitivity map
    #Calculate individual sensitivity map
    ind_grad, _ = sensitivity_fn(inputs_te[insp_epoch:insp_epoch+1,], targets_te[insp_epoch:insp_epoch+1,])
    ind_sm = np.abs(ind_grad)
    ind_sm = np.sum(ind_sm[0],axis=0)
    ind_sm_min=np.min(ind_sm)
    ind_sm_max=np.max(ind_sm)
    ind_sensitivity_map = (ind_sm-ind_sm_min)/(ind_sm_max-ind_sm_min)
    
    #Plot Specific spectrogram
    f = plt.figure()
    extent_v = [-60, 90, 0, 30]
    plt.imshow(np.transpose(inputs_te[insp_epoch],(1,2,0)), interpolation=None, extent=extent_v)
    plt.axes().set_aspect(abs((extent_v[1]-extent_v[0])/(extent_v[3]-extent_v[2])))
    plt.xlabel('time (s)')
    plt.ylabel('frequency (Hz)')
    f.savefig('single_spec_class_'+class_sm+'_sub'+subject_to_evaluate+'.pdf', bbox_inches='tight')
    
    #Plot Specific sensitivity map
    f = plt.figure()
    extent_v = [-60, 90, 0, 30]
    plt.imshow(ind_sensitivity_map, interpolation=None, extent=extent_v)
    plt.axes().set_aspect(abs((extent_v[1]-extent_v[0])/(extent_v[3]-extent_v[2])))
    plt.xlabel('time (s)')
    plt.ylabel('frequency (Hz)')
    f.savefig('single_sm_class_'+class_sm+'_sub'+subject_to_evaluate+'.pdf', bbox_inches='tight')
    