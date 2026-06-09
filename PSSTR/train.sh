#! /usr/bin

source activate
conda activate str
nohup python train.py >> z_output.log 2>&1 &