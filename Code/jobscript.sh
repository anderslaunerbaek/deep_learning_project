#!/bin/sh
### General options: http://www.hpc.dtu.dk/?page_id=2759
### –- specify queue --
#BSUB -q gputitanxpascal
### -- set the job Name --
#BSUB -J testjob
### -- ask for number of cores (default: 4) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 23:59
# request 12GB of memory
#BSUB -R "rusage[mem=12GB]"
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
#BSUB -o gpu-%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --


# Load 
module load python3/3.6.2
source /appl/tensorflow/lsf10-tensorflow-1.4-gpu-python-3.6.2 

cd Documents/Deep_Learning_Project/Code/

python test.py
#python master.py