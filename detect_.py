import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, bbox_iou_, xyxy2xywh_
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def detect(model,
           source = 'inference/images',
           name='exp',
           project = 'runs/detect',
           conf_thres = 0.6,
           iou_thres = 0.45,
           save_txt = False,
           imgsz = 640,
           save_img=True,
           view_img = False,
           exist_ok = False,
           augment = False,
           agnostic_nms = False,
           save_conf = False):
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # # Directories
    if name == '':
        save_dir = Path(project)
    else:
        save_dir = Path(increment_path(Path(project) / name, exist_ok=True))  # increment run
    
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    stride = 64
    if torch.cuda.is_available():
        device = select_device('cuda:0')
        half = True
    else:
        device = select_device('cpu')
        half = False
    
    no_helmet_pos = []
    

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img)[0]#, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img)[0]#, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=agnostic_nms)
        t3 = time_synchronized()
        
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if cls == 0:
                        if len(no_helmet_pos) != 0:
                            idx_to_update = None
                            idx_to_delete = []
                            add = True
                            temp_counter = 0
                            for k, info in enumerate(no_helmet_pos):
                                old_xyxy = info[0]
                                old_frame = info[1]
                                counter = info[2]
                                speed = info[3]
                                
                                old_xywh = xyxy2xywh_(old_xyxy)
                                xywh = xyxy2xywh_(xyxy)
                                time_between = frame - old_frame
                                if time_between == 0:
                                    continue
                 
                                iou = bbox_iou_(old_xyxy,xyxy)
                                if iou >= 0.75:
                                    idx_to_update = k
                                    temp_counter = counter
                                    temp_speed = ((xywh[0]-old_xywh[0])/time_between,(xywh[1]-old_xywh[1])/time_between)
                                    if (temp_speed[0] > 1e+20) or (temp_speed[1] > 1e+20):
                                        temp_speed = None
                                    
                                    elif speed is not None:
                                        new_speed = ((speed[0]+temp_speed[0])/2,(speed[1]+temp_speed[1])/2)
                                        
                                        # cv2.line(im0, (int(xywh[0]),int(xywh[1])), (int(xywh[0]+new_speed[0]),int(xywh[1]+new_speed[1])), (0,0,255), 3)
                                    else:
                                        new_speed = temp_speed
                                    add = False
                                elif speed is not None:
                                    x_pred = speed[0]*time_between
                                    y_pred = speed[1]*time_between
                                    # try:
                                    #     cv2.line(im0, (int(old_xywh[0]),int(old_xywh[1])), (int(old_xywh[0]+x_pred),int(old_xywh[1]+y_pred)), (0,0,255), 3)
                                    # except:
                                    #     print('x_pred:',x_pred)
                                    #     print('y_pred:',y_pred)
                                    
                                    if (abs(old_xywh[0]+x_pred-xywh[0]) < xywh[2]) and (abs(old_xywh[1]+x_pred-xywh[1]) < xywh[3]):
                                        idx_to_update = k
                                        temp_counter = counter
                                        temp_speed = ((xywh[0]-old_xywh[0])/time_between,(xywh[1]-old_xywh[1])/time_between)
                                        new_speed = ((speed[0]+temp_speed[0])/2,(speed[1]+temp_speed[1])/2)
                                        add = False
                                    
                                if frame - old_frame >= 2*dataset.fps: #If the no helmet position hasn't been updated in 30 sec then delete it
                                    idx_to_delete.append(k)
                                    
                            if idx_to_update is not None:
                                no_helmet_pos[idx_to_update] = (xyxy,frame,temp_counter+1,new_speed)
                                print("\n updated position \n")
                                    
                            if len(idx_to_delete) != 0:
                                for index in sorted(idx_to_delete, reverse=True):
                                    del no_helmet_pos[index]
                                    
                            if add:
                                no_helmet_pos.append((xyxy,frame,0,None))
                                print("\n added new position \n")
                        else:
                            no_helmet_pos.append((xyxy,frame,0,None))
                        
                    
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                # Stream results
                
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
                    
            #Save image and timestamp
            for info in no_helmet_pos:
                old_xyxy = info[0]
                old_frame = info[1]
                counter = info[2]
                if (counter == 2*dataset.fps) and (old_frame == frame): # If seen for 2 seconds save an image and timestamp
                    with open(str(save_dir)+'/timestamps.txt', 'a') as fp:
                        fp.write("%s\n" % round(frame/dataset.fps))
                    cv2.imwrite(save_path+'_{}.jpg'.format(frame), im0)
                    print("\n no helmet detected and saved\n")
        
        
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")
                
    print(f'Done. ({time.time() - t0:.3f}s)')