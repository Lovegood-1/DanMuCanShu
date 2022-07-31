from logging import raiseExceptions
import requests
import os 
import json
import torch.backends.cudnn as cudnn 
import torch
import argparse
import time
import yaml
import cv2
import random
# from tqdm import tqdm
import errno
import shutil
from easydict import EasyDict
from termcolor import colored

from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import  select_device, time_synchronized, save_muitl_frame_to_q_k, AutoQueue, event_happend,  Queue_list


@torch.no_grad()
def run(exp_cfg, camera_cfg, root_dir):

    # Initialize
    device = select_device(exp_cfg['device'])
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)  # delete output folder
    os.makedirs(root_dir)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # print(type(device))
    model = attempt_load(exp_cfg['weights'], map_location=device)  # load FP32 model
    imgsz = check_img_size(exp_cfg['image_size'], s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    cudnn.benchmark = True  # set True to speed up constant image size inference
    num_devices = len(camera_cfg)
    dataset = LoadStreams(camera_cfg=camera_cfg, img_size=imgsz, num_devices=num_devices) 

    save_video = False
    T1 = time.perf_counter()
    T_fire = -1
    q_fire =  Queue_list(10) # 用于存储前10帧的火焰bbox
    for img, im0s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=exp_cfg['augment'])[0]

        # Apply NMS
        pred = non_max_suppression(pred, exp_cfg['conf_thres'], exp_cfg['iou_thres'], classes=exp_cfg['classes'], agnostic=exp_cfg['agnostic_nms'])
        t2 = time_synchronized()

        # Process detections
        largest_bbox = [] # to store the largest bbox of fire in current frame
        largest_bbox_area = 0
        for i, det in enumerate(pred):  # detections per image
            T2 =time.perf_counter()
            im0 = im0s[i].copy()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if det is not None and len(det):
                 # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                
                for *xyxy, conf, cls in det:
                    if cls  != 0:
                        continue
                    # calculate the largest bbox
                    wh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()[-2:]
                    if wh[0] * wh [1] > largest_bbox_area:
                        largest_bbox = torch.tensor(xyxy).view(1, 4).tolist()[0]
                    largest_bbox_area = wh[0] * wh [1]

                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
        
        T_fire = time.perf_counter()  

        text_ = "%.1f"%((T2 - T1) *1.0)
        if save_video  == True:
            text_ = 'Saving Video ' + text_
        cv2.putText(im0, text_,(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('p', im0)

        q_fire.auto_put(largest_bbox) # 放每帧的bbox
        # 进入保存视频状态的条件：一个函数，一个变量

              
        if event_happend(q_fire) and save_video == False: # [x]BUG: this 'if' only excute once ! 
            save_video = True
            dataset.event = True
            # T_fire = time.perf_counter() 
        else: 
            save_video = False
            dataset.event = False
        
        save_video=True
        dataset.event=True


















def main():
    parser = argparse.ArgumentParser(description='SimCLR')
    parser.add_argument('--config_env', default='D/edit/doctor_paper/engineering/HS_assesment/code/yolov5/outputs',
                        help='Config file for the environment')
    parser.add_argument('--config_exp', default='./configs/exp_config.yml',
                        help='Config file for the experiment')
    parser.add_argument('--config_camera', default='./configs/camera_config.yml',
                        help='Config file for the camera')
    args = parser.parse_args()
    # Retrieve config file
    exp_cfg = create_config(args.config_exp)
    camera_cfg = create_config(args.config_camera)
    root_dir = args.config_env
    print(colored(exp_cfg, 'yellow'))
    print(colored(camera_cfg, 'cyan'))

    run(exp_cfg, camera_cfg, root_dir)










def create_config(config_file_exp):
    # Config for environment path
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = EasyDict()
   
    # Copy
    for k, v in config.items():
        cfg[k] = v
    return cfg 


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
















if __name__ == '__main__':
    main()
