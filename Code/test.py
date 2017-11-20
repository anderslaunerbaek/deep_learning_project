# coding: utf-8

# # Master notebook

# In[ ]:

from __future__ import absolute_import, division, print_function 
import os
import re
import sys
import time

sys.path.append(os.path.join('.', '..')) 
import utils
import utils_DL
import utils_s160159 as u_s

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import confusion_matrix


# In[ ]:

VERSION = '1.1'
FILENAME = 'master'

print(VERSION)