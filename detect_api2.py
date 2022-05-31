import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import copy
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    build_targets, check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, xywh2xyxy, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized
"""
创建时间：20211202

此文档用于测试 火焰检测 API

主要从 detect.py 复制过来
"""
def detect2(path_video = "inference\\images\\2.mp4,inference\\images\\1.mp4" ,save_img=True):
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
    opt.weights = 'best.pt'
    opt.source = path_video
    opt.save_txt = True



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

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        # save_img = False
        dataset = LoadImages(source, img_size=imgsz)
    # 建立字典，并初始化
    dict_all = {}
    dict_ = {}  
    dict_[0] = {}
    dict_[1] = {}
    num_of_video = len(source.split(','))
    assert num_of_video == 2 # 只支持2个视频
    index_of_video = -1
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
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
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        if dataset.frame in dict_[dataset.count]:
                            dict_[dataset.count][dataset.frame].append(('%g ' * 5 ) % (cls, *xywh))
                        else:
                            dict_[dataset.count][dataset.frame] = []
                            dict_[dataset.count][dataset.frame].append(('%g ' * 5 ) % (cls, *xywh))

                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        # plot_one_box(xyxy, im0, label=None, color=colors[int(cls)], line_thickness=3)  # 只画框，不画类别 置信度
            # Print time (inference + NMS)
            t3 = time.time()
            print('%sDone. (%.3fs); All:(%.3fs)' % (s, t2 - t1, t3 - t1))

            # Stream results
            # if view_img:
            #     cv2.imshow(p, im0)
            #     if cv2.waitKey(1) == ord('q'):  # q to quit
            #         raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        if fps > 100:
                            fps = 30
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
            # print('%sDone.   (%.3fs)' % (s,  time.time() - t1))
        index_of_video = dataset.count
    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    return dict_[0], dict_[1]

def get_max_bbox(dict_):
    pass


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
                detect2()
                strip_optimizer(opt.weights)
        else:
            dict_, dict_2 = detect2("video\\camera_leftrectify.avi,video\\camera_rightrectify.avi")
            # Max_area_bbox = -1
            # Max_area_bbox_index = -1
            # d = dict_
            # new_d_area = {}
            # for num_frame in d:
            #     bbox_list = d[118][0].strip().split(' ')[1:]
            #     bbox =  [float(i) for i in bbox_list]
            #     new_d_area[num_frame] = bbox[2] * bbox[3]
            #     if bbox[2] * bbox[3] > Max_area_bbox:
            #         Max_area_bbox = bbox[2] * bbox[3]
            #         Max_area_bbox_index = num_frame
    sorted_results = sorted(dict_.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    Max_area_bbox_index = sorted_results[1][0]
    # Max_area_bbox = sorted_results[1][-1][0].strip().split(' ')[1:]
    xyhw = [float(i) for i in  sorted_results[1][-1][0].strip().split(' ')[1:] ]
    xyxy=xywh2xyxy(torch.tensor([xyhw]))
    cap = cv2.VideoCapture('video\\camera_leftrectify.avi')
    
    cap.set(cv2.CAP_PROP_POS_FRAMES,Max_area_bbox_index)
    a,b=cap.read()
    gn = torch.tensor(b.shape)[[1, 0, 1, 0]]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(2)]
    xy = [i for i in  xyxy[0]]
    gn_ = [i for i in  gn]
    a_ = xyxy * gn
    plot_one_box([i for i in  a_[0]], b, label=0, color=colors[int(0)], line_thickness=3)
    cv2.imshow('b', b)
    cap.release()
    
    a = 1
    
