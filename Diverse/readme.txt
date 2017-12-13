This piece of code converts EEG data into images to be used in a Convolutional Neural Network to automatically score sleep stages. It can also be used to visualise the reasoning of this network to perform its task. Precise information can be found in the following publication:
https://arxiv.org/abs/1710.00633

The current directory contains:
- readme.txt -> The current file
- eeg_to_image.m -> Matlab file that converts Physionet EEG sleep signals to RGB images.
- eeg_vgg_sleep.py -> Python file that trains a VGG net to work with EEG sleep data.
- eeg_vgg_sleep_sensitivity.py -> Python file that computes sensitivity maps on the pretrained VGG net for EEG sleep scoring.

If you want to use the code, there are a set of dependencies that your computer needs to fulfil:

- Sleep EEG Data: The data we use is from PhysioNet (https://www.physionet.org/) repository. Specifically, a dataset called “sleep-edfx”, which contains polysomnographies (PSG’s) from healthy people:
https://www.physionet.org/physiobank/database/sleep-edfx
If you do not want to manually download the files from the site, you can use the following Matlab toolbox to do so:
https://github.com/anasimtiaz/sleep-edfx-toolbox

- To run eeg_to_image.m you need to have Matlab installed in your machine, together with the Chronux-Toolbox: http://chronux.org 

- To run eeg_vgg_sleep.py and eeg_vgg_sleep_sensitivity.py, you need to have Python installed, together with the last version of Theano and Lasagne libraries.
Moreover, if you want to use network weights pre-trained on the ImageNet challenge, you will need to download the weights in a pickle file from: https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

Hence, the structure of the directory should look like:
+eeg_vgg
  |—- readme.txt
  |—- eeg_to_image.m
  |—— eeg_vgg_sleep.py
  |—— eeg_vgg_sleep_sensitivity.py
  |—— vgg16.pkl
  +—— outputs
  +—— PhysioNet_Sleep_EEG
      +—— subX_nX_img_fpz
          |—— img_XX.png
          |—— labels.txt

Please, cite the work as: 
Title:	"Deep Convolutional Neural Networks for Interpretable Analysis of EEG Sleep Stage Scoring"
Authors: Albert Vilamala, Kristoffer H. Madsen, Lars K. Hansen. 
Publication: eprint arXiv:1710.00633 
url: https://arxiv.org/abs/1710.00633

For more information, or any bug you might find, please contact: 
Albert Vilamala
alvmu@dtu.dk
http://people.compute.dtu.dk/alvmu 
DTU Compute, Technical University of Denmark. 