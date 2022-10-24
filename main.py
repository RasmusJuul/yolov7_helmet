import argparse
from detect_ import detect
from utils.torch_utils import TracedModel
from utils.general import increment_path
from models.experimental import attempt_load
import os
import torch
import cv2
from pathlib import Path

def split_video(vidSource, outputPath):
    vidPath = vidSource
    shotsPath = outputPath

    cap = cv2.VideoCapture(vidPath)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    segRange = [(0,300*fps),
                (300*fps,300*fps*2),
                (300*fps*2,300*fps*3),
                (300*fps*3,300*fps*4),
                (300*fps*4,300*fps*5),
                (300*fps*5,300*fps*6),
                (300*fps*6,300*fps*7),
                (300*fps*7,300*fps*8),
                (300*fps*8,300*fps*9),
                (300*fps*9,300*fps*10),
                (300*fps*10,300*fps*11),
                (300*fps*11,300*fps*12)] # a list of starting/ending frame indices pairs

    for idx,(begFidx,endFidx) in enumerate(segRange):
        writer = cv2.VideoWriter(shotsPath%idx,fourcc,fps,size)
        cap.set(cv2.CAP_PROP_POS_FRAMES,begFidx)
        ret = True # has frame returned
        while(cap.isOpened() and ret and writer.isOpened()):
            ret, frame = cap.read()
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            if frame_number < endFidx:
                writer.write(frame)
            else:
                break
        writer.release()
        
        

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='inference/videos/20220928_142037.mp4', help='path to video')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    opt = parser.parse_args()
    
    project = "20220928" #Date
    name = "14" #Hour
    
    save_dir = Path(increment_path(Path(project) / name, exist_ok=False))
    (save_dir / 'videos').mkdir(parents=True, exist_ok=True)
    
    split_video(opt.source, str(save_dir.absolute())+'/videos/%d.mp4')

    device = 'cuda:0'

    # Load model
    imgsz = 640
    model = attempt_load('best_yolov7.pt', map_location=device)  # load FP32 model
    model = TracedModel(model, device, imgsz)
    model.half()  # to FP16
    
    for i in range(12):
        source = str(save_dir.absolute())+'/videos/{}.mp4'.format(i)
        detect(model, source = source, name = str(5+5*i), project = '{}/{}'.format(project,name))


