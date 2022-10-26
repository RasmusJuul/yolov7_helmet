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
    
    vid_length = 300

    segRange = [(0,vid_length*fps),
                (vid_length*fps,vid_length*fps*2),
                (vid_length*fps*2,vid_length*fps*3),
                (vid_length*fps*3,vid_length*fps*4),
                (vid_length*fps*4,vid_length*fps*5),
                (vid_length*fps*5,vid_length*fps*6),
                (vid_length*fps*6,vid_length*fps*7),
                (vid_length*fps*7,vid_length*fps*8),
                (vid_length*fps*8,vid_length*fps*9),
                (vid_length*fps*9,vid_length*fps*10),
                (vid_length*fps*10,vid_length*fps*11),
                (vid_length*fps*11,vid_length*fps*12)] # a list of starting/ending frame indices pairs

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
    parser.add_argument('--date', default='2022/0928', help='save results to date/hour')
    parser.add_argument('--hour', default='14', help='save results to date/hour')
    opt = parser.parse_args()
    
    date = opt.date
    hour = opt.hour
    
    save_dir = Path(increment_path(Path(date) / hour, exist_ok=False))
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
        detect(model, source = source, name = str(5+5*i), project = '{}/{}'.format(date,hour))


