#!/bin/sh
#BSUB -q gpuv100
#BSUB -J yolov7_test
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -R "rusage[mem=8GB]"
##BSUB -R "select[gpu40gb]" #options gpu40gb or gpu80gb
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/gpu_%J.err
# -- end of LSF options --

nvidia-smi

module load python3
module load cuda

source ../yolov7env/bin/activate

python test.py --data data/custom.yaml --img 448 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights best.pt --name yolov7-helmet_test