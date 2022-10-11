from detect_ import detect
from utils.torch_utils import TracedModel
from models.experimental import attempt_load
from utils.general import check_img_size
import os

device = 'cuda:0'

# Load model
imgsz = 640
model = attempt_load('best.pt', map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size
model = TracedModel(model, device, imgsz)
model.half()  # to FP16

detect(model, source = "inference/videos/test.mp4",name="202210111829")


n_frames = len(os.listdir("runs/detect/202210111829/labels"))

consecutive = 0
when = []

for i in range(1,n_frames+1):
    is_no_helmet = False
    with open("runs/detect/202210111829/labels/test_{}.txt".format(i), 'r') as file:
        bbs = file.readlines()
    for bb in bbs:
        if bb[0] == '0':
            is_no_helmet = True
    if is_no_helmet:
        consecutive += 1
    else:
        consecutive = 0
    if consecutive == 30:
        when.append(int(i/30))