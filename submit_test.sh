#!/bin/sh
#BSUB -q gpuv100
#BSUB -J yolov7_test
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -R "rusage[mem=8GB]"
##BSUB -R "select[gpu40gb]" #options gpu40gb or gpu80gb
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/gpu_%J.err
# -- end of LSF options --

nvidia-smi

module load python3
module load cuda

source ../yolov7env/bin/activate

python main.py --source inference/videos/paving.mp4
