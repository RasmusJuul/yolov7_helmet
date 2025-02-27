#!/bin/sh
#BSUB -q gpua100
#BSUB -J yolov7_test
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 01:00
#BSUB -R "rusage[mem=8GB]"
##BSUB -R "select[gpu40gb]" #options gpu40gb or gpu80gb
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/gpu_%J.err
# -- end of LSF options --

nvidia-smi

module load python3
module load cuda

source ../yolov7env/bin/activate

python test.py --data data/custom.yaml --img 640 --batch 32 --conf 0.5 --iou 0.65 --device 0 --weights yolov7-beanie.pt --name yolov7_640_val

# python test.py --data data/custom.yaml --img 640 --batch 32 --conf 0.5 --iou 0.65 --device 0 --weights yolov7-beanie.pt --name yolov7-e6e_640_val