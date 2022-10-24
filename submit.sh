#!/bin/sh
#BSUB -q gpua100
#BSUB -J yolov7
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=8GB]"
##BSUB -R "select[gpu40gb]" #options gpu40gb or gpu80gb
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/gpu_%J.err
# -- end of LSF options --

nvidia-smi

module load cuda

source ../yolov7env/bin/activate

#python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7-e6e-custom.yaml --weights 'yolov7-e6e_training.pt' --name yolov7-helmet --hyp data/hyp.scratch.custom.yaml
python train.py --workers 8 --device 0 --batch-size 16 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7-custom.yaml --weights 'yolov7.pt' --name yolov7-helmet-basic --hyp data/hyp.scratch.custom.yaml
