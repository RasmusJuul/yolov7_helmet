#!/bin/sh
#BSUB -q gpua100
#BSUB -J yolov7_detect
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 03:00
#BSUB -R "rusage[mem=8GB]"
##BSUB -R "select[gpu40gb]" #options gpu40gb or gpu80gb
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/gpu_%J.err
# -- end of LSF options --

nvidia-smi

module load python3
module load cuda

source ../yolov7env/bin/activate

# python detect.py --weights best_yolov7-e6e.pt --conf 0.70 --img-size 640 --source inference/videos/test.mp4 --name 'e6e'

python detect.py --weights yolov7-beanie.pt --conf 0.70 --img-size 640 --source inference/videos/test.mp4 --name 'basic'
