#!/bin/sh
### General options: http://www.hpc.dtu.dk/?page_id=2759
### -- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J sensitivity_v1
### -- ask for number of cores (default: 1) --
#BSUB -n 24
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 72:00
#BSUB -R "rusage[mem=10GB] span[hosts=1]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o tensorflow_v4_sen-cpu-%J.out
##BSUB -e tensorflow_v4_sen-cpu-%J.err
# -- end of LSF options --

# Load
module load python3/3.6.2
module load scipy/0.19.1-python-3.6.2
module load matplotlib/2.0.2-python-3.6.2
source /appl/tensorflow/1.4cpu-python362/bin/activate

cd ~/Documents/deep/Code/ 

python3 sensitivity.py
