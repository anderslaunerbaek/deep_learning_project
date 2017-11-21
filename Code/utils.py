# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:27:26 2016

@author: alvmu
"""

import numpy as np

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    batchsize = int(batchsize)
    last_idx = int(batchsize*(len(inputs)/batchsize))
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    #for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
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

        # return as generator
        yield inputs[excerpt], targets[excerpt]