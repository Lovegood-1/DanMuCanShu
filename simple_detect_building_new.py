import argparse
import os
import socket
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import PIL
import matplotlib # 注意这个也要import一次
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from io import BytesIO
import cv2
import torch
from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import (LOGGER, check_img_size, check_requirements, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
import math
import json
import warnings
warnings.filterwarnings('ignore')

import ddrnet_23_slim
import numpy as np
import time
from tqdm import tqdm

HOST='127.0.0.1'
PORT=8885
plt.style.use("seaborn-dark")
plt.rcParams['savefig.dpi'] = 200 #图片像素
plt.rcParams['figure.dpi'] = 200 #分辨率
plt.rcParams['figure.figsize'] = (10.8, 7.2) # 设置figure_size尺寸
plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False

"""
--------------------yolov5--------------------
"""
# inference
@torch.no_grad()
def run_det(weights='yolov5s.pt',  # model.pt path(s)
        source='source',  # file/dir/URL/glob, 0 for webcam
        data='data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device=0,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    # params
    source = str(source)
    det_list = []
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()
    elif engine and model.trt_fp16_input != half:
        LOGGER.info('model ' + (
            'requires' if model.trt_fp16_input else 'incompatible with') + ' --half. Adjusting automatically.')
        half = model.trt_fp16_input
    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    seen =  0
    for file_path, im, im0s, _, _ in dataset:
        if '/' in file_path: 
            file_name = file_path.split('/')[-1]
        else:
            file_name = file_path.split('\\')[-1]
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        pred = model(im, augment=augment)
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # Process predictions
        for det in pred:  # per image
            det_per_image = []
            seen += 1
            im0 = im0s.copy()
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    det_per_image.append({'label':names[c], 'conf':round(conf.item(),2), 'bbox':[int(val.item()) for val in xyxy]})
            # Stream results
            im0 = annotator.result()
            # Save results (image with detections)
            if not nosave:
                cv2.imwrite(os.path.join(save_path, 'det_'+file_name), im0)
            det_list.append(det_per_image)
    return det_list

    
# set parameters        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=None, help='model path(s)')
    parser.add_argument('--source', type=str, default='inputs/building', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--nosave', default=True, action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt



"""
--------------------HsSegNet--------------------
"""
def input_transform(image):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

def run_seg(source, save=False):
    # load segmentation model
    device = select_device(0)
    model=ddrnet_23_slim.get_final_model(pretrained=False).to(device)
    pretrained_dict = torch.load('weights/seg_building_best.pth')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()

    # read image
    files = sorted(os.listdir(source))[:2]
    hs_img = cv2.imread(os.path.join(source, files[-1]))
    # hs_img = cv2.resize(hs_img,(2160,1440),interpolation=cv2.INTER_NEAREST)
    image = hs_img.copy()
    image=input_transform(image)
    image = image.transpose((2, 0, 1))
    img = np.expand_dims(image,0)
    im = torch.from_numpy(img).to(device)
    with torch.no_grad():
        outputs = model(im)
    pred = torch.argmax(outputs[1],1)
    pred = pred.cpu().data.numpy()
    predict = pred.squeeze(0)
    maskr = cv2.resize(predict,(0,0),fx=8,fy=8,interpolation=cv2.INTER_NEAREST)

    # save segment result to image
    if save:
        empty_img = np.zeros_like(hs_img)
        empty_img[...,1] = np.where(maskr==1,255,empty_img[...,1])
        empty_img[...,2] = np.where(maskr==2,255,empty_img[...,2])
        new_img = cv2.add(empty_img, hs_img)
        cv2.imwrite(os.path.join(source, 'seg_'+files[-1]), new_img)
    
    return maskr




"""
--------------------post processing--------------------
"""
# capture two images from a video, save in source
def cap_images_to_source(video_path, source):
    cap = cv2.VideoCapture(video_path)
    frame_start = int(cap.get(0))
    frame_end = int(cap.get(7))
    # print(frame_start, frame_end)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    _, img_start = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_end-5)
    _, img_end = cap.read()
    cap.release()
    cv2.imwrite(os.path.join(source, "frame_a.jpg"), img_start)
    cv2.imwrite(os.path.join(source, "frame_b.jpg"), img_end)


# resize the images in source directory into shape: (2160, 1440)
def resize_source(source):
    files = sorted(os.listdir(source))[:2]
    for file in files:
        img = cv2.imread(os.path.join(source, file))
        img = cv2.resize(img,(2160,1440),interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(source, file), img)


# check if the two bboxes are overlap 
def is_overlap(boxa, boxb):
    ax1, ay1, ax2, ay2 = boxa
    bx1, by1, bx2, by2 = boxb

    rx = abs((ax1+ax2)/2 - (bx1+bx2)/2)
    ry = abs((ay1+ay2)/2 - (by1+by2)/2)
    wa, ha = ax2-ax1, ay2-ay1
    wb, hb = bx2-bx1, by2-by1

    if rx <= (wa+wb)/2 and ry <= (ha+hb)/2:
        return True
    else:
        return False


# calculate IoU
def iou(boxa, boxb):
    ax1, ay1, ax2, ay2 = boxa
    bx1, by1, bx2, by2 = boxb
    sa = (ax2-ax1) * (ay2-ay1)
    sb = (bx2-bx1) * (by2-by1)
    X = min(ax2, bx2) - max(ax1, bx1)
    Y = min(ay2, by2) - max(ay1, by1)
    S = X * Y
    iou = S / (sa+sb-S)
    return iou


# if det has the object, return the index, else return -1
def exist_obj(det, label):
    for i, obj in enumerate(det):
        if obj['label'] == label:
            return i
    return -1


# for every object in supression_list, limit the num to one by area. (save max area object)
# supression_list: (default) ['head', 'carriage', 'launcher']
def nms_by_area(det_list, supression_list):
    nms_det_list = []
    for det in det_list:
        nms_det = []
        for obj in det:
            i = exist_obj(nms_det, obj['label'])
            if obj['label'] not in supression_list or i == -1:
                nms_det.append(obj)
            else:
                new_s = (obj['bbox'][2]-obj['bbox'][0]) * (obj['bbox'][3]-obj['bbox'][1])
                cur_s = (nms_det[i]['bbox'][2]-nms_det[i]['bbox'][0]) * (nms_det[i]['bbox'][3]-nms_det[i]['bbox'][1])
                if new_s > cur_s: 
                    nms_det[i] = obj 
        nms_det_list.append(nms_det)
    return nms_det_list


# if the obj overlap with objs in the list, return the index, else return -1
def is_overlap_with_nms_det(obj, nms_det, supression, t):
    for i, item in enumerate(nms_det):
        if item['label'] == supression and is_overlap(obj['bbox'], item['bbox']) and iou(obj['bbox'], item['bbox']) > t:
            return i
    return -1


# supress the repetition bbox of supression
def nms_by_iou(det_list, supression_list, t=0.5):
    for supression in supression_list:
        nms_det_list = []
        for det in det_list:
            nms_det = []
            for obj in det:
                if obj['label'] != supression:
                    nms_det.append(obj)
                else:
                    i = is_overlap_with_nms_det(obj, nms_det, supression, t)
                    if i == -1: 
                        nms_det.append(obj)
                    else:
                        new_s = (obj['bbox'][2]-obj['bbox'][0]) * (obj['bbox'][3]-obj['bbox'][1])
                        cur_s = (nms_det[i]['bbox'][2]-nms_det[i]['bbox'][0]) * (nms_det[i]['bbox'][3]-nms_det[i]['bbox'][1])
                        if new_s < cur_s: 
                            nms_det[i] = obj
            nms_det_list.append(nms_det)
        det_list = nms_det_list
    return det_list


# find all_damage and part_damage objects, save them in list
def match_by_iou(det_list, t=0):
    det_a, det_b = det_list[0:2]
    all_damage_objs = []
    part_damage_objs = []
    for obj_a in det_a:
        finded = False
        for obj_b in det_b:
            if obj_a['label'] == obj_b['label'] and is_overlap(obj_a['bbox'], obj_b['bbox']) \
            and iou(obj_a['bbox'], obj_b['bbox']) >= t: 
                part_damage_objs.append([obj_a, obj_b])
                finded = True
                break
        # can not find a bbox from obj_b that matches with current bbox in a, so all damage
        if not finded:
            all_damage_objs.append(obj_a)
    return all_damage_objs, part_damage_objs


# find all_damage and part_damage objects, save them in the list 
def match_by_label(det_list, label_list):
    det_a, det_b = det_list[0:2]
    all_damage_objs = [] 
    part_damage_objs = []
    for obj_a in det_a:
        # for objects in label_list, match by label
        if obj_a['label'] in label_list:
            finded = False
            for obj_b in det_b:
                if obj_a['label'] == obj_b['label']:
                    part_damage_objs.append([obj_a, obj_b])
                    finded = True
                    break
            if not finded:
                all_damage_objs.append(obj_a)
    # if object not in label_list, append them in part_damage_objs
    for obj_b in det_b:
        if obj_b['label'] not in label_list:
            part_damage_objs.append([obj_b, obj_b])
    return all_damage_objs, part_damage_objs
                

# english to chinese
def en2ch(label):
    dicts = {
        'person': '人','bicycle': '自行车','car': '汽车','motorcycle': '摩托车','airplane': '飞机','bus': '公共汽车','train': '火车','truck': '卡车','boat': '船','traffic light': '红绿灯',
        'fire hydrant': '消防栓','stop sign': '停车标志','parking meter': '停车计','bench': '台阶','bird': '鸟','cat': '猫','dog': '狗','horse': '马','sheep': '羊','cow': '牛',
        'elephant': '大象','bear': '熊','zebra': '斑马','giraffe': '长颈鹿','backpack': '背包','umbrella': '雨伞','handbag': '手提包','tie': '领带', 'suitcase': '行李箱',
        'frisbee': '飞筒','skis': '滑雪板','snowboard': '雪板','sports ball': '运动球','kite': '风筝','baseball bat': '棒球棒','baseball glove': '棒球帽','skateboard': '冲浪板','surfboard': '冲浪板',
        'tennis racket': '网球拍','bottle': '瓶子','wine glass': '酒杯','cup': '杯子','fork': '叉子','knife': '刀','spoon': '匙','bowl': '碗','banana': '香蕉','apple': '苹果',
        'sandwich': '三明治','orange': '橘子','broccoli': '西兰花','carrot': '胡萝卜','hot dog': '热狗','pizza': '比萨','donut': '甜甜圈','cake': '蛋糕','chair': '椅子','couch': '沙发',
        'potted plant': '盆栽','bed': '床','dining table': '餐桌','toilet': '卫生间','tv': '电视','laptop': '笔记本电脑','mouse': '鼠标','remote': '遥控器','keyboard': '键盘','cell phone': '手机',
        'microwave': '微波炉','oven': '烤箱','toaster': '烤面包机','sink': '水槽','refrigerator': '冰箱','book': '书','clock': '钟','vase': '花瓶','scissors': '剪刀','teddy bear': '玩具熊',
        'hair drier': '吹风机','toothbrush': '牙刷', 'building':'建筑', 'window':'窗户'
    }
    return dicts[label] if label in dicts.keys() else '未知'


# to support chinese in opencv
def cv2AddChineseText(img, text, position, size, color, font="simhei.ttf"):
    r, g, b = color
    color = (b, g, r)
    font = ImageFont.truetype(font, size, encoding="utf-8")
    if (isinstance(img, np.ndarray)): 
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    draw.text(position, text, color, font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# img: cv2 image
def show_bbox(img, det, color, chinese=True, ignore=None, fill=None, hide_labels=None, limit_bboxes=None):
    for obj in det:
        label = obj['label']
        # check whether it is in the list that is not displayed
        if not(ignore and label in ignore):
            x1, y1, x2, y2 = obj['bbox']
            show = True
            # check whether the bbox to be displayed is within the legal limit_bboxes
            if limit_bboxes:
                for bbox in limit_bboxes:
                    if x1 < bbox[0] or y1 < bbox[1] or x2 > bbox[2] or y2 > bbox[3]:
                        show = False
                        break
            # if True, show bbox
            if show:
                thickness = -1 if fill and label in fill else 3     # whether to fill
                img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                if not(hide_labels and label in hide_labels):
                    if chinese:  
                        img = cv2AddChineseText(img, en2ch(label), (x1, y1-55), 55, color)
                    else: 
                        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX_SMALL,3, color)
    return img



# get bbox by label
def get_bbox_by_label(det, label):
    bbox = []
    for obj in det:
        if obj['label'] == label:
            bbox = obj['bbox']
            break
    return bbox


# write title in top left
def write_title(img, text, text_size, color_bg, color_title):
    blank = 10
    ret_x = text_size * len(text) + blank
    ret_y = text_size + blank
    cv2.rectangle(img, (0,0), (ret_x,ret_y), color_bg, thickness=-1)
    img = cv2AddChineseText(img, text, (blank//2,blank//2), text_size, color_title)
    return img


# get bgr color
def bgr_color():
    return {
        'green' :   (  0, 255,   0),
        'blue'  :   (255, 191,   0),
        'red'   :   (  0,   0, 255),
        'yellow':   (  0, 255, 255),
        'white' :   (255, 255, 255),
        'black' :   (  0,   0,   0),
        'orange':   (0, 165,   255),
        'gray'  :   (105, 105, 105)}
    

# # draw landing point
# def draw_point(im, det, point, colors, HW=(50, 70)):
#     H, W = HW 
#     bbox = get_bbox_by_label(det, label='building')
#     x0, y0 = bbox[0], bbox[3]
#     xlen, ylen = bbox[2]-bbox[0], bbox[3]-bbox[1]
#     # draw origin
#     im = cv2.circle(im, (x0, y0), radius=10, color=colors['blue'], thickness=20)
#     # draw axis
#     im = cv2.line(im, (x0, y0), (x0, y0-ylen), color=colors['blue'], thickness=10)
#     im = cv2.line(im, (x0, y0), (x0+xlen, y0), color=colors['blue'], thickness=10)
#     # draw arrow
#     arrow = 50
#     sin = int(arrow * math.sin(math.pi/6))
#     cos = int(arrow * math.cos(math.pi/6))
#     im = cv2.line(im, (x0-sin, y0-ylen+cos), (x0, y0-ylen), color=colors['blue'], thickness=10)
#     im = cv2.line(im, (x0+sin, y0-ylen+cos), (x0, y0-ylen), color=colors['blue'], thickness=10)
#     im = cv2.line(im, (x0+xlen-cos, y0-sin), (x0+xlen, y0), color=colors['blue'], thickness=10)
#     im = cv2.line(im, (x0+xlen-cos, y0+sin), (x0+xlen, y0), color=colors['blue'], thickness=10)
#     # draw landing point
#     im = cv2.circle(im, point, radius=20, color=colors['red'], thickness=40)
#     # add text
#     text_size = 60
#     im = cv2AddChineseText(im, '高', (x0-sin-text_size, y0-ylen+10), text_size, colors['blue'])
#     im = cv2AddChineseText(im, '宽', (x0+xlen, y0+sin), text_size, colors['blue'])
#     im = cv2AddChineseText(im, str(H), (x0-sin-text_size, y0-ylen+10+2*text_size), text_size, colors['black'])
#     im = cv2AddChineseText(im, str(W).format(70), (x0+xlen-2*text_size, y0+sin), text_size, colors['black'])
#     im = cv2AddChineseText(im, '落点', (point[0]+text_size, point[0]-text_size), text_size, colors['red'])
#     return im


# draw landing point
def draw_point(im, det, point, colors, HW):
    colors = bgr_color()
    H, W = HW
    text_size = 60
    im_h, im_w = im.shape[:2]
    start_h = 100
    start_w = 250
    end_h = im_h - start_h
    end_w = im_w - start_w
    floor_h = end_h - start_h
    floor_w = end_w - start_w
    room_h = int((floor_h) / H)
    room_w = int((floor_w) / W)
    # draw grad
    for i in range(W+1):
        x1 = start_w + room_w * i
        x2 = x1
        y1 = start_h
        y2 = end_h
        im = cv2.line(im, (x1, y1), (x2, y2), color=colors['black'], thickness=2)
        if i < W:
            im = cv2AddChineseText(im, str(i+1), (x1+int(room_w/2-text_size/2), y2+int(text_size/3)), text_size, colors['black'])
    for i in range(H+1):
        x1 = start_w
        x2 = end_w
        y1 = start_h + room_h * i 
        y2 = y1
        im = cv2.line(im, (x1, y1), (x2, y2), color=colors['black'], thickness=2)
        if i < H:
            n = 3/2 if len(str(H-i)) == 2 else 1
            im = cv2AddChineseText(im, str(H-i), ((x1-int(n*text_size)), y1+int(room_h/2-text_size/2)), text_size, colors['black'])
        
    # draw axis
    bias = 5
    im = cv2.line(im, (start_w, start_h), (start_w, end_h-bias), color=colors['blue'], thickness=12)
    im = cv2.line(im, (start_w, end_h-bias), (end_w, end_h-bias), color=colors['blue'], thickness=12)
    # draw landing point
    for i, pt in enumerate(point):
        floor, room = pt
        pt = start_w + int((room-0.5) * room_w), start_h + int(((H-floor)+0.5) * room_h)
        im = cv2.circle(im, pt, radius=int(0.2*min(room_h, room_w)), color=colors['red'], thickness=int(0.4*min(room_h, room_w)))
        n = 2.2 if len(str(i+1)) == 2 else 1
        pt = pt[0]-int(0.2*n*text_size), pt[1]-int(0.5*text_size)
        im = cv2AddChineseText(im, '{}'.format(i+1), pt, text_size, colors['black'])
    # draw arrow
    arrow = 40
    sin = int(arrow * math.sin(math.pi/6))
    cos = int(arrow * math.cos(math.pi/6))
    im = cv2.line(im, (start_w-sin, start_h+cos), (start_w, start_h), color=colors['blue'], thickness=10)
    im = cv2.line(im, (start_w+sin, start_h+cos), (start_w, start_h), color=colors['blue'], thickness=10)
    im = cv2.line(im, (end_w-cos, end_h-sin-bias), (end_w, end_h-bias), color=colors['blue'], thickness=10)
    im = cv2.line(im, (end_w-cos, end_h+sin-bias), (end_w, end_h-bias), color=colors['blue'], thickness=10)
    # add text
    im = cv2AddChineseText(im, '层数', (start_w-sin-int(3.2*text_size), start_h), text_size, colors['blue'])
    im = cv2AddChineseText(im, '间数', (end_w, end_h+sin), text_size, colors['blue'])
    return im


# get the damage rate of all cameras on each floors
def get_floors(camera_damage):
    floors = {}
    for camera in camera_damage:
        location = camera['location']
        damage = camera['damage']
        floor_name = '{}'.format(location[0]) 
        area_id = '{}{}'.format(location[1], location[2])
        if  floor_name not in floors.keys():
            floors[floor_name] = {}
        floors[floor_name][area_id] = damage['overall_rate_conf'][0]
    return floors


# get bbox of each camera
def create_camera_bboxes(floors, floors_bbox, text_size):
    # set value
    cam_h = text_size + int(text_size/2)    # if 'simhe.ttc'
    cam_w = int(3.4*text_size)              # if 'simhe.ttc'
    # cam_h = text_size + int(text_size/4)  # if 'msyhbd.ttc'
    # cam_w = int(3.2*text_size)            # if 'msyhbd.ttc'
    cam_pad_w = 40
    cam_pad_h = 20
    camera_bboxes = {}
    for floor_name in floors:
        camera_bboxes[floor_name] = {}
        floor = floors[floor_name]
        floor_bbox = floors_bbox[floor_name]
        floor_w = floor_bbox[2] - floor_bbox[0]
        floor_h = floor_bbox[3] - floor_bbox[1]
        floor_i = floor_w // (cam_w + cam_pad_w)
        start_w = floor_bbox[0]
        start_h = floor_bbox[1]
        j = 0
        for i, area in enumerate(floor):
            # x1, x2
            x1 = start_w + (cam_w + cam_pad_w) * (i % floor_i) + cam_pad_w
            x2 = x1 + cam_w
            # update j
            j = i // floor_i
            # y1, y2
            y1 = start_h + (cam_h + cam_pad_h) * j + cam_pad_h
            y2 = y1 + cam_h
            bbox = x1, y1, x2, y2
            camera_bboxes[floor_name][area] = [bbox, floor[area]]
    return camera_bboxes


# convert rate to color
def rate2color(rate):
    if rate < 0:
        color = 'gray'
    elif rate >= 0 and rate <= 0.3:
        color = 'green'
    elif rate > 0.3 and rate <= 0.6:
        color = 'orange'
    else:
        color = 'red'
    return color


# convert area to text
def area2text(area):
    text = ''
    if area[0] == 'R':
        text += '房间'
    elif area[0] == 'B':
        text += '室外'
    elif area[0] == 'L':
        text += '走廊'
    else:
        text += '未知'
    text += area[1:]
    return text


# draw bbox of each camera
def draw_camera_bboxes(img4, camera_bboxes, floors_bbox, colors, text_size):
    for floor_name in camera_bboxes:
        floor_bbox = floors_bbox[floor_name]
        floor_w = floor_bbox[2] - floor_bbox[0]
        floor_h = floor_bbox[3] - floor_bbox[1]
        floor_cameras = camera_bboxes[floor_name]
        for area in floor_cameras:
            bbox = [int(v) for v in floor_cameras[area][0]]
            rate = floor_cameras[area][1]
            if bbox[3] <= floor_bbox[3]:
                img4 = cv2.rectangle(img4, bbox[:2], bbox[2:4], colors[rate2color(rate)], thickness=-1)
                # p_text = [v+int(text_size/8) for v in bbox[:2]]   # if 'simhe.ttc'
                p_text = [v+int(text_size/12) for v in bbox[:2]]    # if 'msyhbd.ttc'
                img4 = cv2AddChineseText(img4, '{}'.format(area2text(area)), p_text, text_size, colors['white'], font='msyhbd.ttc')
            else:
                p_text = [int(floor_bbox[2] - 1.4*text_size), int(floor_bbox[3] - floor_h/2 - text_size/2)]
                img4 = cv2AddChineseText(img4, '...', p_text, text_size, colors['black'], font='msyhbd.ttc')
                break
    return img4


# show status of every camera
def show_camera(img4, camera_damage, colors):
    # get floors
    floors = get_floors(camera_damage)
    # set value
    im_h, im_w = img4.shape[:2]
    start_h = 100
    start_w = 50
    end_h = im_h-start_h
    end_w = im_w-start_w
    text_size = 55
    text_pad = 20
    thickness = 3
    # draw bbox of floors
    floors_bbox = {}
    img4 = cv2.rectangle(img4, (start_w, start_h), (end_w, end_h), colors['black'], thickness=thickness)
    floor_h = int((end_h-start_h) / len(floors.keys()))
    for i, key in enumerate(sorted([int(k) for k in floors.keys()], reverse=True)): 
        line_h = (i+1) * floor_h
        text_h = line_h - int(floor_h/2) - int(text_size/2)
        img4 = cv2AddChineseText(img4, '{}层相机'.format(key), (start_w+text_pad, start_h+text_h), text_size, colors['black'])
        if i+1 != len(floors):
            img4 = cv2.line(img4, (start_w, start_h+line_h), (end_w, start_h+line_h), colors['black'], thickness=thickness)
        floors_bbox[str(key)] = [start_w+text_pad+4*text_size, start_h+line_h-floor_h, end_w, start_h+line_h]
    img4 = cv2.line(img4, (start_w+text_pad+4*text_size, start_h), (start_w+text_pad+4*text_size, end_h), colors['black'], thickness=thickness)
    # get bbox of each camera
    text_size = 50  # if 'simhe.ttc' delete
    camera_bboxes = create_camera_bboxes(floors, floors_bbox, text_size)
    # draw patch of each camera
    img4 = draw_camera_bboxes(img4, camera_bboxes, floors_bbox, colors, text_size)
    return img4
    

# visualize in damaged image
def visualize_outdoor(det_list, save_path, camera_damage, point, HW=(12,6)):
    colors = bgr_color()
    # read img1, img2
    img_files = sorted(os.listdir(save_path))
    img1 = cv2.imread(os.path.join(save_path, img_files[0]))
    img2 = cv2.imread(os.path.join(save_path, img_files[1]))
    img3 = np.ones_like(img1) * 255
    img4 = np.ones_like(img1) * 255

    # show detected objects on images
    building_bbox_1 =  get_bbox_by_label(det_list[0], 'building')
    building_bbox_2 =  get_bbox_by_label(det_list[1], 'building')
    img1 = show_bbox(img1, det_list[0], colors['green'], hide_labels=['window'], limit_bboxes=[building_bbox_1])
    img2 = show_bbox(img2, det_list[1], colors['green'], hide_labels=['window'], limit_bboxes=[building_bbox_2])

    # write title on images
    img1 = write_title(img1, text="毁伤前", text_size=80, color_bg=colors['gray'], color_title=colors['white'])
    img2 = write_title(img2, text="毁伤后", text_size=80, color_bg=colors['gray'], color_title=colors['white'])
    img3 = write_title(img3, text="落点估计", text_size=80, color_bg=colors['gray'], color_title=colors['white'])
    img4 = write_title(img4, text="室内毁伤", text_size=80, color_bg=colors['gray'], color_title=colors['white'])
    
    # draw point
    img3 = draw_point(img3, det_list[1], point, colors, HW)
    # show camera with damage damage
    img4 = show_camera(img4, camera_damage, colors)
    # draw landing point
    # concatenate img1, img2, img3, img4
    joint12 = np.concatenate((img1, img2), axis=1)
    joint34 = np.concatenate((img3, img4), axis=1)
    joint = np.concatenate((joint12, joint34), axis=0)
    # write image
    cv2.imwrite(os.path.join(save_path, 'c.png'), joint)


# get color_mask
def get_color_mask(mask, img, colors):
    color_mask = np.zeros_like(img)
    mask = np.transpose(np.array([mask, mask, mask]), (1,2,0))
    # color_mask = np.where(mask==1, colors['green'], color_mask)
    color_mask = np.where(mask==2, colors['red'], color_mask)
    return color_mask


# visualize in damaged image
def visualize_indoor(det_list, save_path, mask=None):
    colors = bgr_color()
    # read img1, img2
    img_files = sorted(os.listdir(save_path))
    img1 = cv2.imread(os.path.join(save_path, img_files[0]))
    img2 = cv2.imread(os.path.join(save_path, img_files[1]))
    img3 = img1.copy()
    img4 = img2.copy()

    # show detected objects on images
    img3 = show_bbox(img3, det_list[0], color=colors['green'])
    img4 = show_bbox(img4, det_list[1], color=colors['green'])

    # write title on images
    img1 = write_title(img1, text="毁伤前原图", text_size=80, color_bg=colors['gray'], color_title=colors['white'])
    img2 = write_title(img2, text="毁伤后原图", text_size=80, color_bg=colors['gray'], color_title=colors['white'])
    img3 = write_title(img3, text="毁伤前检测", text_size=80, color_bg=colors['gray'], color_title=colors['white'])
    img4 = write_title(img4, text="毁伤后检测", text_size=80, color_bg=colors['gray'], color_title=colors['white'])
    
    # show segment on img4
    if mask is not None:
        color_mask = get_color_mask(mask, img4, colors)
        img4 = cv2.add(img4, color_mask, dtype=cv2.CV_8U)
    
    # concatenate img1, img2, img3, img4
    joint12 = np.concatenate((img1, img2), axis=1)
    joint34 = np.concatenate((img3, img4), axis=1)
    joint = np.concatenate((joint12, joint34), axis=0)
    # write image
    cv2.imwrite(os.path.join(save_path, 'c.png'), joint)


def get_damage(det_list, mask=None):
    cate_num_a = {}     # {'category': num} for image a
    cate_num_b = {}     # {'category': num} for image b
    cate_confs = {}     # {'category': [conf_1, conf_2]} for all cate
    # for image a
    for obj in det_list[0]:
        label = obj['label']
        conf  = obj['conf']
        if label not in cate_num_a.keys():
            cate_num_a[label] = 1
            cate_confs[label] = [conf]
        else:
            cate_num_a[label] += 1
            cate_confs[label].append(conf)
    # for image b
    for obj in det_list[1]:
        label = obj['label']
        conf  = obj['conf']
        if label not in cate_num_b.keys():
            cate_num_b[label] = 1
            cate_confs[label] = [conf]
        else:
            cate_num_b[label] += 1
            cate_confs[label].append(conf)
    # calculate damage and conf
    cate_rate = {}
    cate_nums = {}
    overall_rate = []
    overall_conf = []
    for cate in cate_num_a:
        num_a = cate_num_a[cate]
        num_b = cate_num_b[cate] if cate in cate_num_b else 0
        rate = (num_a - num_b) / num_a if num_a != 0 else 0
        if rate > 1:
            rate = 1
        conf = np.asarray(cate_confs[cate]).mean()
        cate_rate[cate] = (rate, conf)
        cate_nums[cate] = (num_a, num_b)
        overall_rate.append(rate)
        overall_conf.append(conf)
    overall_rate = np.asarray(overall_rate).mean() if len(overall_rate) != 0 else 0
    overall_conf = np.asarray(overall_conf).mean() if len(overall_conf) != 0 else 0
    # segment damage rate
    if mask is not None:
        weight = {'num':0.8, 'seg':0.2}
        try:
            seg_rate = len(np.where(mask==2)[0]) / len(np.where(mask!=2)[0])
        except:
            seg_rate = 0
        overall_rate = weight['num'] * overall_rate + weight['seg'] * seg_rate
    damage = {'overall_rate_conf':(overall_rate, overall_conf), 'rate_conf':cate_rate, 'objs_nums':cate_nums}
    return damage


def infer_point(camera_damage, y_coord):
    # initialize point 
    point = [-1, -1]
    # get floor_rate for each floors
    floors = get_floors(camera_damage)
    floors_rate = {}
    for floor_name in floors:
        floor = floors[floor_name]
        floors_rate[floor_name] = 0
        for area in floor:
            floors_rate[floor_name] += abs(floor[area])
    # initialize max_floor & get max damaged floor
    max_floor = {'rate':0, 'name':'0'}
    if y_coord: # if the max damage floor is known
        max_floor = {'rate':floors_rate[str(y_coord)], 'name':str(y_coord)}
    else:       # if not, search for the maximum damage floor
        for floor_name in floors_rate:
            if floors_rate[floor_name] > max_floor['rate']:
                max_floor['rate'] = floors_rate[floor_name]
                max_floor['name'] = floor_name
    # initialize max_area & get max dameged room
    max_area = {'rate':0, 'name':'0'}
    if max_floor['name'] != '0':
        floor = floors[max_floor['name']]
        for area in floor:
            area_rate = floor[area]
            if area_rate > max_area['rate']:
                max_area['rate'] = area_rate
                max_area['name'] = area
        if max_area['name'] != '0':
            point = [int(max_floor['name']), int(max_area['name'][1:])]
    return point

    

"""
--------------------function entry--------------------
"""

if __name__ == '__main__':
    # read camera_info_paths
    with open('camera/camera_info_paths.txt', 'r') as f:
        paths = f.readlines()
    # read camera_info_path one by one from paths
    for i, camera_info_path in enumerate(paths):
        # read all cameras in camera_info_path
        with open(camera_info_path.strip('\n'), 'r') as f:
            camera_info = json.load(f)
        print('>>> 第{:<2}/{:<2}发'.format(i+1, len(paths)))
        # collect cameras by area
        camera_b_list = []  # save outdoor cameras
        camera_l_list = []  # save zoulang cameras
        camera_r_list = []  # save room cameras    
        camera_damage = []  # collect all camera parameters and corresponding area damage information  
        for camera in camera_info:
            if camera['location'][1] == 'B':
                camera_b_list.append(camera)
            if camera['location'][1] == 'L':
                camera_l_list.append(camera)
            if camera['location'][1] == 'R':
                camera_r_list.append(camera)
        # ---------------------------------------------- #
        # -----------Indoor Assessment Image------------ #
        # ---------------------------------------------- #
        print('>>> 室内评估 ...')
        # read each camera in camera_r_list
        for camera in tqdm(camera_r_list):
            # save outdoor cameras
            if camera['location'][1] == 'B':
                camera_b_list.append(camera)
                continue
            if camera['location'][1] == 'L':
                camera_l_list.append('')
            # get save_path of camera
            save_path = camera['save_path']
            # if save_path exists image a and b, camera works
            files = sorted(os.listdir(save_path))[:2]
            if len(files) == 2 and 'a' in files[0].split('.')[0] and 'b' in files[1].split('.')[0]:
                # resize images
                resize_source(save_path)
                # run yolov5 and get detected object of per image (two images)
                opt = parse_opt()
                opt.source = save_path
                opt.weights = 'yolov5x6.pt'
                det_list = run_det(**vars(opt))
                # supress the given objects by area, choose max area
                det_list = nms_by_area(det_list, supression_list=['building'])
                # supress the given objects by iou, choose min area
                det_list = nms_by_iou(det_list, supression_list=['window'], t=0.1)
                # run HsSegNet
                seg = True
                mask = run_seg(save_path, save=False) if seg else None
                # visualize on indoor images
                visualize_indoor(det_list, save_path, mask)
                # get damage
                damage = get_damage(det_list, mask)
                camera['state'] = 'work'
            else:   # if save_path does not exist image a and b, camera failed
                damage = {'overall_rate_conf':(-1, -1), 'rate_conf':None, 'objs_nums':None}
                camera['state'] = 'fail'
            # add damage dict into camera
            camera['damage'] = damage
            camera_damage.append(camera)
            # save each camera
            with open(os.path.join(save_path, 'e.json'), 'w') as f:
                json.dump(camera, f)
        # save camera_damage
        camera_damage_path = os.path.join(os.path.split(camera_info_path)[0], 'camera_damage.json') 
        with open(camera_damage_path, 'w') as f:
            json.dump(camera_damage, f)
        # # ---------------------------------------------- #
        # # -----------Outdoor Assessment Video----------- #
        # # ---------------------------------------------- #
        # for camera in tqdm(camera_b_list):
        #     save_path = camera['save_path']

        # ---------------------------------------------- #
        # -----------Outdoor Assessment Image----------- #
        # ---------------------------------------------- #
        print('>>> 室外评估 ...')
        # read y_coord of landing point
        y_coord_path = os.path.join(os.path.split(camera_info_path)[0], 'y_coord.json')
        try:
            with open(y_coord_path, 'r') as f:
                y_coord = json.load(f)
            y_coord = y_coord[0] if len(y_coord) == 1 else None
        except:
            print('y_coord is none, enable reverse inference.')
            y_coord = None
        # read each camera in camera_b_list
        for camera in tqdm(camera_b_list):
            save_path = camera['save_path']
            # resize images
            resize_source(save_path)
            # run yolov5 and get detected object of per image (two images)
            opt = parse_opt()
            opt.source = save_path
            opt.weights = 'weights/det_building_best.pt'
            det_list = run_det(**vars(opt))
            # supress the given objects by area, choose max area
            det_list = nms_by_area(det_list, supression_list=['building'])
            # supress the given objects by iou, choose min area
            det_list = nms_by_iou(det_list, supression_list=['window'], t=0.1)
            # infer landing point and save
            point = infer_point(camera_damage, y_coord)
            point_path = os.path.join(camera_info_path.split('\\')[0], 'point.json')
            with open(point_path, 'w') as f:
                json.dump(point, f)
            # visualize on outdoor images
            visualize_outdoor(det_list, save_path, camera_damage, [point], HW=(12,24))

    # waiting for close
    print(">>> 评估结束")