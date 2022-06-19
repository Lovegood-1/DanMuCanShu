import math
import os
import time
import logging
from copy import deepcopy
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# from utils.general import xywh2xyxy
import  calibration.param as stereoconfig_040_2
from utils.video_rectify import video_rectify
import random


logger = logging.getLogger(__name__)
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
def init_seeds(seed=0):
    torch.manual_seed(seed)

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            logger.info("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        logger.info('Using CPU')

    logger.info('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # Finds layer indices matching module class 'mclass'
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    # Return global model sparsity
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent
    print(' %.3g global sparsity' % sparsity(model))


def fuse_conv_and_bn(conv, bn):
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    with torch.no_grad():
        # init
        fusedconv = nn.Conv2d(conv.in_channels,
                              conv.out_channels,
                              kernel_size=conv.kernel_size,
                              stride=conv.stride,
                              padding=conv.padding,
                              bias=True).to(conv.weight.device)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv


def model_info(model, verbose=False):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        flops = profile(deepcopy(model), inputs=(torch.zeros(1, 3, 64, 64),), verbose=False)[0] / 1E9 * 2
        fs = ', %.1f GFLOPS' % (flops * 100)  # 640x640 FLOPS
    except:
        fs = ''

    logger.info('Model Summary: %g layers, %g parameters, %g gradients%s' % (len(list(model.parameters())), n_p, n_g, fs))


def load_classifier(name='resnet101', n=2):
    # Loads a pretrained model reshaped to n-class output
    model = models.__dict__[name](pretrained=True)

    # Display model properties
    input_size = [3, 224, 224]
    input_space = 'RGB'
    input_range = [0, 1]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for x in [input_size, input_space, input_range, mean, std]:
        print(x + ' =', eval(x))

    # Reshape output to n classes
    filters = model.fc.weight.shape[1]
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model


def scale_img(img, ratio=1.0, same_shape=False):  # img(16,3,256,416), r=ratio
    # scales img(bs,3,y,x) by ratio
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            gs = 32  # (pixels) grid size
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


import queue

def build_multi_queue(seconds, fps, names = ['rgb', 'left', 'right'], type = 'normal'):
    """
    
    """
    if type == 'list':
        _build_queue = build_list_queue
    else:
        _build_queue = build_queue
    q_d = {}
    for name in names:
        q_d[name] = _build_queue(seconds, fps)
    return q_d


def build_queue(seconds, fps):
    """
    This F use in  saving a video of the period prior to a certain moment.
 
    """
    lengh_ = seconds * fps * 2
    q = queue.Queue(int(lengh_))
    return q

def build_list_queue(seconds, fps):
    """
    This F use in  saving a serise of bbox .
 
    """
    lengh_ = seconds * fps * 2
    q = Queue_list(int(lengh_))
    return q


def save_muitl_frame_to_q_k(q_k, frames, key, names = ['rgb_video', 'left', 'right']):
    for name in names:
        q = save_frame_to_q(q_k, frames[name], key, name)
        q_k[name] = q

    return q_k


def save_frame_to_q(q, frame, key, name):
    """save to queue. This F use in  saving a video of the period prior to a certain moment.
    if is full?
    |- Ture : drop the 1st and put 
    |- False: put 
    """
    q_k = q[key][name]
    if q_k.qsize() < q_k.maxsize:
        q_k.put(frame)
    else:
        q_k.get()
        q_k.put(frame)
    q[key][name] = q_k
    return q

import cv2
def save_multi_video(q,fourcc, FPS, size, names = ['rgb', 'left', 'right'], device = 'usb', save_video_path = 'video'):

    if device != 'OTA':
        out_ = { i: cv2.VideoWriter(os.path.join(save_video_path,'camera_%s.avi'%i), fourcc, FPS, (size[0]//2,size[1]))    for i in names }
    else:
        out_ = { i: cv2.VideoWriter('camera_%s.avi'%i, fourcc, FPS, size) if 'rgb' in i else  cv2.VideoWriter('camera_%s.avi'%i, fourcc, FPS, (1280,720), isColor=False)  for i in names }
    # print('camera_%s.avi'%i)
    for name in names:
        while True:
            if q['before'][name].qsize()>0:
                out_[name].write(q['before'][name].get())
            else:
                break
        while True:
            if q['after'][name].qsize()>0:
                out_[name].write(q['after'][name].get())
            else:
                break        
        out_[name].release()

class Queue_list(object):
    def __init__(self, maxlen):               #初始化空队列
        self.list = []
        self.maxlen = maxlen
 
    def _put(self,item):             #入队
        self.list.append(item)       #尾进
        # self.list.insert(0,item)   #头进
 
    def get(self):                   #出队
        # return self.list.pop(0)    #头出
         return self.list.pop(0)      #尾出
 
    def is_empty(self):              # 判断是否为空
        return self.list == []
 
    def qsize(self):                  # 判断长度
        return len(self.list)
 
    def __str__(self):               #遍历所有队列当中的队员
        return "队员(%r)" % self.list

    def auto_put(self, element):
        """
        add a element into the top of queue as usual.
        But if it is full, delete the first one and add the final
        """
        if len(self.list) < self.maxlen:
            self._put(element)
        else:
            self.get()
            self._put(element)

class AutoQueue:
    def __init__(self, maxitem) -> None:
        import queue
        self.q = queue.Queue(int(maxitem))

    def auto_put(self, element):
        """
        add a element into the top of queue as usual.
        But if it is full, delete the first one and add the final
        """
        if self.q.qsize() < self.q.maxsize:
            self.q.put(element)
        else:
            self.q.get()
            self.q.put(element)

def event_happend(q_list):

    happend = True
    for frame in q_list.list:
        if len(frame) == 0:
            happend = False
    return happend


def cal_xyz_from_uv(x,y,disp,Q):
    """
    从像素坐标计算三维坐标
    """
    deep =  disp[int(y), int(x)] #
    vec_tmp = np.array([[x,y,deep,1]]).T
    vec_tmp = Q@ vec_tmp
    vec_tmp /= vec_tmp[3]
    return vec_tmp

def cal_distance(p1,p2,vertical = False):
    return math.sqrt(math.pow(p2[0] - p1[0],2) + math.pow(p2[1] - p1[1],2) + math.pow(p2[2] - p1[2],2))



def show_point(uv_list,img):
    img = img.copy()
    index = 0
    for i in uv_list:
        cv2.circle(img, (int(i[1]), int(i[2])),1,(0,0,255),-1)
        print(int(i[1]), int(i[2]))
        index += 1
        cv2.putText(img,"%s"%str(index), (int(i[1]), int(i[2])), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,205,100), 1, cv2.LINE_AA)
    return img

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def calculate_bbox_area_from_list(bbox_list):
    """_summary_

    Args:
        bbox_list (list): fire info in every frame 
            ['0, x, y, h, w', '0, x, y, h, w', ...]
    """
    max_area  = -1
    max_bbox = []
    for bbox_ in bbox_list:
        bbox = bbox_.strip().split(' ')[1:]
        bbox = [float(i.strip()) for i in bbox]
        if bbox[-2] * bbox[-1] > max_area:
            max_bbox = bbox
            max_area = bbox[-2] * bbox[-1]
    return max_area, max_bbox 


def _2xyxy(right_bbox, img_shape):
    """convert xywh -> xyxy(左上右下)

    Args:
        right_bbox (_type_): _description_
        frame (_type_): _description_

    Returns:
        _type_: _description_
    """
    center_x = right_bbox[0] * img_shape[1]
    center_y = right_bbox[1] * img_shape[0]
    w = right_bbox[2] * img_shape[1]
    h = right_bbox[3] * img_shape[0]
    tl_x = center_x-w/2
    tl_y = center_y-h/2
    br_x = center_x+w/2
    br_y = center_y+h/2
    l = [tl_x, tl_y, br_x, br_y]
    l = [int(i) for i in  l]
    return l

def show_bbox(path_, max_bbox_index, left_bbox):
    cap = cv2.VideoCapture(path_) 
    cap.set(cv2.CAP_PROP_POS_FRAMES,max_bbox_index)
    a,imgl=cap.read()
    cap.release()
    
    imgl = cv2.rectangle(imgl, (left_bbox[0],left_bbox[1]), (left_bbox[2],left_bbox[3]), (0,255,0), 4)
    return imgl

def get_max_bbox(path, dict_, dict2_):
    """在左相机中找最大bbox

    Args:
        dict_ (dict[narray]): fire info in every frame
            {num_frame: ['0, x, y, h, w', '0, x, y, h, w', ...]}
    Returns:
        _type_: _description_
        fire_left: the pixel coordinates of bbox left bottom in both camera
            (4,) for a bbox:[xl,yl,xr,yr]
        fire_right: the pixel coordinates of bbox right bottom in both camera
            (4,) for a bbox:[xl,yl,xr,yr]
        pixel_distance_of_high: the pixel distance of the height of bbox in left camera
            (1,) for a bbox  
        bbox: the largetest bbox
            (narray) for a bbox 
        frame: the index of the largetest bbox
            (int)
    """
    cap = cv2.VideoCapture(path.split(',')[0])
    ret, frame = cap.read()
    cap.release()
    max_bbox_area = -1
    max_bbox_index = -1
    left_bbox = [] # xyhw
    right_bbox = []
    for frame_l in dict_:
        # 如果是当前最大，判断又相机满足：1 单个火焰，则记录下来
        bbox_area, bbox = calculate_bbox_area_from_list(dict_[frame_l])
        if bbox_area > max_bbox_area:
            if frame_l in dict2_:
                if len(dict2_[frame_l]) == 1:
                    max_bbox_area = bbox_area
                    max_bbox_index = frame_l
                    left_bbox = bbox
                    _, bbox = calculate_bbox_area_from_list(dict2_[frame_l])
                    right_bbox = bbox
    # 转化为像素坐标，选择性可视化
    left_bbox = _2xyxy(left_bbox, frame.shape)
    right_bbox = _2xyxy(right_bbox, frame.shape)
    fire_left = [left_bbox[0], left_bbox[3], right_bbox[0], right_bbox[3],]
    fire_right = [left_bbox[2], left_bbox[3], right_bbox[2], right_bbox[3],]
    pixel_distance_of_high = left_bbox[3] - left_bbox[1]
    if 1 == 1:
        # 获取指定帧，左右相机的图片,用于验证bbox是否正确
        imgl_bbox = show_bbox(path.split(',')[0], max_bbox_index, left_bbox)
        imgr_bbox = show_bbox(path.split(',')[-1], max_bbox_index, right_bbox)
        cv2.imshow('left',imgl_bbox)
        cv2.imshow('right', imgr_bbox)

    return fire_left, fire_right, pixel_distance_of_high, left_bbox, max_bbox_index
   

def show_video(video_path, window_name, times = 1):
    frame_counter = 0
    video_path = video_path.split('.')[0] + '_results.mp4'
    assert os.path.isfile(video_path)
    cap = cv2.VideoCapture(video_path)
    play_times = 0
    # 2.循环读取图片
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow(window_name, frame)
        else:
            cap = cv2.VideoCapture(video_path)
            
            if play_times >= times:
                print("视频播放完成！")
                break
            if times == 1:
                break
            play_times += 1
        # 退出播放
        key = cv2.waitKey(25)
        if key == 27:  # 按键esc
            break
 


if __name__ == "__main__":
    q = Queue_list(4)
    a = 1