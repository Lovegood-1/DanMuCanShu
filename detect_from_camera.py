import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import  LoadImages, LoadStreams_OTA
from utils.datasets import LoadStreams_double_usb as LoadStreams
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized, build_queue, save_frame_to_q, build_multi_queue,save_muitl_frame_to_q_k,save_multi_video, AutoQueue, event_happend,  Queue_list

def detect(save_img=False, camera_device = 'usb', save_video_path = 'video'):
    names_video = ['rgb_video', 'left', 'right'] if camera_device == 'OTA' else ['left', 'right']
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

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

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # some values initailization
    seconds = 3
    save_video = False
    FPS = 10
    q = {'before': build_multi_queue(seconds , fps = FPS, names  = names_video  ), 'after':build_multi_queue(seconds = 3, fps = FPS, names  = names_video ) }
    # q_bbox =  {'before': build_multi_queue(seconds , fps = FPS, names  = names_video , type='list' ), 'after':build_multi_queue(seconds = 3, fps = FPS, names  = names_video , type='list') }
    T1 = time.perf_counter()
    T_fire = -1
    q_fire =  Queue_list(10) # 用于存储前10帧的火焰bbox

    # NEW: DEVICE SETTIING
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams_OTA(source, img_size=imgsz) if camera_device == 'OTA' else LoadStreams(source, img_size=imgsz) 
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)
    size = dataset.size
    total_largest_bbox_area = 0
    Max_file_img = {}
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

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        largest_bbox = [] # to store the largest bbox of fire in current frame
        largest_bbox_area = -1
        for i, det in enumerate(pred):  # detections per image
            T2 =time.perf_counter()
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

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

        if largest_bbox_area != -1 and largest_bbox_area > total_largest_bbox_area:  # 有火焰且大于目前的
            total_largest_bbox_area = largest_bbox_area
            Max_file_img['img'] = {'left': vid_cap['left'].copy(), 'right': vid_cap['right'].copy()}
            Max_file_img['bbox'] = {'xyxy': largest_bbox}
            # Stream results
        if view_img:
            text_ = "%.1f"%((T2 - T1) *1.0)

            if save_video  == True:
                text_ = 'Saving Video ' + text_
            cv2.putText(im0, text_,(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(p, np.concatenate((im0, vid_cap['right'].copy()), axis = 1))
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration
            if cv2.waitKey(1) & 0xFF == ord('s'): # press s to force save video
                save_video = True
                T_fire = time.perf_counter()
                print('sssssssssssssssss')
        q_fire.auto_put(largest_bbox) # 放每帧的bbox,共10帧
        if event_happend(q_fire) and save_video == False: # [x]BUG: this 'if' only excute once ! 
            save_video = True
            T_fire = time.perf_counter()  
        if 1 == 1:
            
            if camera_device == 'OTA':
                vid_cap['rgb_video'] = im0
            if T_fire != -1: #  only save after fire
                if (T2 - T_fire) < seconds:
                    q = save_muitl_frame_to_q_k(q, vid_cap, 'after', names_video)
                else:
                    break
            else:
                q = save_muitl_frame_to_q_k(q, vid_cap, 'before', names_video)
                
    
    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)
    
    save_multi_video(q, fourcc, FPS, size, names_video, save_video_path)
    
    
    time.sleep(0.5)
    # ======== 后处理开始 ===========

    # 1 视频矫正
    # 2 深度图估计
    # ==============================
    import  calibration.param as stereoconfig_040_2
    from utils.video_rectify import video_rectify

    # 1 视频矫正
    print('视频矫正')
    left_video = r'video\camera_left.avi'
    right_video = r'video\camera_right.avi'
    config = stereoconfig_040_2.stereoCamera()
    Video = video_rectify(left_video, right_video, config)
    Video.rectify_video(left_video, 'left')
    Video.rectify_video(right_video, 'right')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
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
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
            # import  calibration.param as stereoconfig_040_2
            # from utils.video_rectify import video_rectify
            # print('视频矫正')
            # left_video = r'video\camera_left.avi'
            # right_video = r'video\camera_right.avi'
            # config = stereoconfig_040_2.stereoCamera()
            # Video = video_rectify(left_video, right_video, config)
            # Video.rectify_video_double(left_video, 'left', right_video, 'right')
            # Video.rectify_video(right_video, 'right')
