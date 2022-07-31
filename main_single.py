from glob import glob
import cv2
from flask import Flask, render_template, Response, request
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import threading
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import Load_RTSP_cameras
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import interval_happend, select_device, load_classifier, time_synchronized, build_queue, save_frame_to_q, build_multi_queue,save_muitl_frame_to_q_k,save_multi_video, AutoQueue, event_happend,  Queue_list

"""
无多线程
"""
app = Flask(__name__)
# 1 定义全局变量 停止；触发事件；
#               模型；相机（dataset）；超参数
global stop_, force_save, model, dataset, opt, device ,half, names, colors,  save_video
model, dataset, opt = -1, -1, -1
save_video = False
global grey
grey = False
def init():
    """初始化全局变量
       此函数只能运行一遍！因为有删除保存目录的功能
    """
    global model, dataset, opt, device, half , names, colors
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference\\output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    # 设置设备{0，rtsp文件}
    source = 0
    dataset = Load_RTSP_cameras(source, img_size=imgsz) 

def yolo_gen():
    global model, dataset, opt, device , half, names, colors, save_video

    
    T1 = time.perf_counter()
    view_img = True
    T_fire = -1
    q_fire =  Queue_list(10) # 用于存储前10帧的火焰bbox
    for path, img, im0s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        largest_bbox = [] # to store the largest bbox of fire in current frame
        largest_bbox_area = 0

        for i, det in enumerate(pred):  # detections per image
            T2 =time.perf_counter()
            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
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

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
        T_fire = time.perf_counter()  

        # Stream results
        if view_img:
            text_ = "%.1f"%((T2 - T1) *1.0)
            if save_video  == True or grey:
                text_ = 'Saving Video ' + text_
            cv2.putText(im0, text_,(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # cv2.imshow(p, im0)
            image = cv2.imencode('.jpg', im0)[1].tobytes()
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
        q_fire.auto_put(largest_bbox) # 放每帧的bbox

        # 进入保存视频状态的条件：一个函数，一个变量
        if event_happend(q_fire) and save_video == False: # [x]BUG: this 'if' only excute once ! 
            save_video = True
            # T_fire = time.perf_counter()  

        # 如果在保存视频，但是已经10f没有火焰了，则认为间隔发生
        if  interval_happend(q_fire):
            save_video = False


def gen(save_img=False, camera_device = 'usb'):
    global stop_
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference\\output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    dataset = LoadStreams(source, img_size=imgsz, rtsp=rtsp) 
    # some values initailization
    save_video = False
    T1 = time.perf_counter()
    T_fire = -1
    q_fire =  Queue_list(10) # 用于存储前10帧的火焰bbox
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        largest_bbox = [] # to store the largest bbox of fire in current frame
        largest_bbox_area = 0
        for i, det in enumerate(pred):  # detections per image
            T2 =time.perf_counter()
            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if cls  != 0:
                        continue
                    # calculate the largest bbox
                    wh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()[-2:]
                    if wh[0] * wh [1] > largest_bbox_area:
                        largest_bbox = torch.tensor(xyxy).view(1, 4).tolist()[0]
                    largest_bbox_area = wh[0] * wh [1]
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
        T_fire = time.perf_counter()  

        # Stream results
        if view_img:
            text_ = "%.1f"%((T2 - T1) *1.0)
            if save_video  == True:
                text_ = 'Saving Video ' + text_
            cv2.putText(im0, text_,(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # cv2.imshow(p, im0)
            image = cv2.imencode('.jpg', im0)[1].tobytes()
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
        q_fire.auto_put(largest_bbox) # 放每帧的bbox

        # 进入保存视频状态的条件：一个函数，一个变量
        if event_happend(q_fire) and save_video == False: # [x]BUG: this 'if' only excute once ! 
            save_video = True
            # T_fire = time.perf_counter()  

        # 如果在保存视频，但是已经10f没有火焰了，则认为间隔发生
        if  interval_happend(q_fire):
            save_video = False
        
                
    
    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)
    
    time.sleep(2)
    print('Done. (%.3fs)' % (time.time() - t0))
    stop_ = True


@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/video_feed')
def video_feed():
    return Response(yolo_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey
        elif  request.form.get('neg') == 'Negative':
            global neg
            neg=not neg
        elif  request.form.get('face') == 'Face Only':
            global face
            face=not face 
            if(face):
                time.sleep(4)   
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
                          
                 
    elif request.method=='GET':
        return render_template('index2.html')
    return render_template('index2.html')


init()

app.run()


