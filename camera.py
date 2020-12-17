import time
import multiprocess as mp
import ctypes
from dlclivegui.queue import ClearableQueue, ClearableMPQueue
import threading
import cv2
import numpy as np
import os
from dlclive import DLCLive, Processor
import math

cap = cv2.VideoCapture('/home/diya/Documents/dlclive/short.avi')
dlc_proc = Processor()
dlc_live = DLCLive('/home/diya/Documents/dlclive/thursday-diya-2020-10-22/exported-models/DLC_thursday_resnet_50_iteration-0_shuffle-1', processor=dlc_proc)
fps=100
ret, frame = cap.read()
start_time = time.time()

pose_list = []
frame_list = []
while frame is not None:
    dlc_live.init_inference(frame)
    pose_list.append(dlc_live.get_pose(frame))
    current_time = time.time()
    current_frame = math.trunc((current_time - start_time) * fps)
    print(current_frame)
    if current_frame > int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        break
    else:
        cap.set(cv2.CAP_PROP_FRAME_COUNT, current_frame)
        ret, frame = cap.read()

print(pose_list)


