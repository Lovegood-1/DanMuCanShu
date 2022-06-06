 # -*- coding: utf-8 -*-
from traceback import print_tb
import cv2
from cv2 import CALIB_CB_ACCURACY
from cv2 import CALIB_ZERO_DISPARITY
import numpy as np
import  calibration.param_0531 as stereoconfig_040_2   #导入相机标定的参数
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from models import *
import cv2
from PIL import Image
import math
# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/
def pars():
    parser = argparse.ArgumentParser(description='PSMNet')
    parser.add_argument('--KITTI', default='2015',
                        help='KITTI version')
    parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                        help='select model')
    parser.add_argument('--loadmodel', default='pretrained_model_KITTI2015.tar',
                        help='loading model')
    parser.add_argument('--leftimg', default= './VO04_L.png',
                        help='load model')
    parser.add_argument('--rightimg', default= './VO04_R.png',
                        help='load model')                                      
    parser.add_argument('--model', default='RTStereoNet',
                        help='select model')
    parser.add_argument('--maxdisp', type=int, default=192,
                        help='maxium disparity')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    global args
    global model
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.model == 'stackhourglass':
        model = stackhourglass(args.maxdisp)
        model = nn.DataParallel(model)
    elif args.model == 'basic':
        model = basic(args.maxdisp)
        model = nn.DataParallel(model)    
    elif args.model == 'RTStereoNet':
        model = RTStereoNet(args.maxdisp)
    else:
        print('no model')

    if args.cuda:
        model.cuda()

    if args.loadmodel is not None:
        print('load model')
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'])

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    return args, model
def test(imgL,imgR):
        model.eval()

        if args.cuda:
           imgL = imgL.cuda()
           imgR = imgR.cuda()     

        with torch.no_grad():
            disp = model(imgL,imgR)

        disp = torch.squeeze(disp)
        pred_disp = disp.data.cpu().numpy()

        return pred_disp


def main(path_l,path_r):
    pars()
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(**normal_mean_var)])    

    imgL_o = Image.open(path_l).convert('RGB')
    imgR_o = Image.open(path_r).convert('RGB')

    imgL = infer_transform(imgL_o)
    imgR = infer_transform(imgR_o) 
    

    # pad to width and hight to 16 times
    if imgL.shape[1] % 16 != 0:
        times = imgL.shape[1]//16       
        top_pad = (times+1)*16 -imgL.shape[1]
    else:
        top_pad = 0

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16                       
        right_pad = (times+1)*16-imgL.shape[2]
    else:
        right_pad = 0    

    imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
    imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

    start_time = time.time()
    pred_disp = test(imgL,imgR)
    print('time = %.2f' %(time.time() - start_time))

    
    if top_pad !=0 and right_pad != 0:
        img = pred_disp[top_pad:,:-right_pad]
    elif top_pad ==0 and right_pad != 0:
        img = pred_disp[:,:-right_pad]
    elif top_pad !=0 and right_pad == 0:
        img = pred_disp[top_pad:,:]
    else:
        img = pred_disp
    return img

# 预处理
def preprocess(img1, img2):
    # 彩色图->灰度图
    if(img1.ndim == 3):#判断为三维数组
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
    if(img2.ndim == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 直方图均衡
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)

    return img1, img2


# 消除畸变
def undistortion(image, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)

    return undistortion_image


# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
# @param：config是一个类，存储着双目标定的参数:config = stereoconfig.stereoCamera()
def getRectifyTransform(height, width, config):
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    # 计算校正变换
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion, 
                                                    (width, height), R, T, alpha=1, flags=CALIB_ZERO_DISPARITY)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion, 
                                                    (width, height), R, T, alpha=0)
    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q

# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)

    return rectifyed_img1, rectifyed_img2


# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    return output


# 视差计算
def stereoMatchSGBM(left_image, right_image, down_scale=False, up_scale = False):
    # SGBM匹配参数设置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 5
    paraml = {'minDisparity': 1,
             'numDisparities': 64,
             'blockSize': blockSize,
             'P1': 8 * img_channels * blockSize ** 2,
             'P2': 32 * img_channels * blockSize ** 2,
             'disp12MaxDiff': 1,
             'preFilterCap': 63,
             'uniquenessRatio': 15,
             'speckleWindowSize': 150,
             'speckleRange': 1,
             'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
             }

    # 构建SGBM对象
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)

    # 计算视差图
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)

    elif down_scale == True:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]

        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right
    if up_scale == True:
        left_image_down = cv2.pyrUp(left_image)
        right_image_down = cv2.pyrUp(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]

        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right        
    # 真实视差（因为SGBM算法得到的视差是×16的）
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.

    return trueDisp_left, trueDisp_right


# 将h×w×3数组转换为N×3的数组
def hw3ToN3(points):
    height, width = points.shape[0:2]

    points_1 = points[:, :, 0].reshape(height * width, 1)
    points_2 = points[:, :, 1].reshape(height * width, 1)
    points_3 = points[:, :, 2].reshape(height * width, 1)

    points_ = np.hstack((points_1, points_2, points_3))

    return points_

class point():
    def __init__(self) -> None:
        self.p0 = [0,0]

    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pass

def cal_distance(p1,p2):
    return math.sqrt(math.pow(p2[0] - p1[0],2) + math.pow(p2[1] - p1[1],2) + math.pow(p2[2] - p1[2],2))

def cal_xyz(x,y,deep,Q):
    vec_tmp = np.array([[x,y,deep,1]]).T
    vec_tmp = Q@vec_tmp
    vec_tmp /= vec_tmp[3]
    return vec_tmp[0], vec_tmp[1], vec_tmp[2]

def print_function(dis, last_point, points_3d, x, y):
    # 是否有 inf 在
    if (math.isinf(points_3d[y, x, 0]) or math.isinf(points_3d[y, x, 1]) ) or math.isinf(points_3d[y, x, 2]) :
        print("INF!",points_3d[y, x, 0], points_3d[y, x, 1], points_3d[y, x, 2])
        return 0 
    if (math.isinf(last_point[0]) or math.isinf(last_point[1]) ) or math.isinf(last_point[2]) :
        print("INF!",last_point[0] ,last_point[1] ,last_point[2] )
        return 0     
    dis_ = cal_distance(last_point,[points_3d[y, x, 0], points_3d[y, x, 1], points_3d[y, x, 2]]) # 深度 + Q ； 两点距离
    
    print('点 (%0.3f, %0.3f, %0.3f)与点 (%0.3f, %0.3f, %0.3f)的距离为%0.3f m' % (last_point[0], last_point[1], last_point[2], points_3d_dl[y, x, 0], points_3d_dl[y, x, 1], points_3d_dl[y, x, 2], dis_))
    # 
    pass

if __name__ == '__main__':

    i = 3
    string = 're'
    # 读取数据集的图片
    iml = cv2.imread(r'data\0601\WIN_20220601_08_57_15_Pro.jpgleft.png', 1)  # 左图
    imr = cv2.imread(r'data\0601\WIN_20220601_08_57_15_Pro.jpgright.png', 1)  # 右图
    height, width = iml.shape[0:2]

    print("width = %d \n"  % width)
    print("height = %d \n" % height)
    

    # 读取相机内参和外参
    config = stereoconfig_040_2.stereoCamera()

    # 立体校正
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
    iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)

    print("Print Q!")
    print(Q)

    # 绘制等间距平行线，检查立体校正的效果
    line_rec = draw_line(iml_rectified, imr_rectified)
    line = draw_line(iml, imr)
    cv2.imwrite('depth_0531_17_07.png' ,  np.concatenate((line, line_rec), axis = 0))

    iml_rectified_l, imr_rectified_r = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
    cv2.imwrite('templ.png', iml_rectified_l)
    cv2.imwrite('tempr.png', imr_rectified_r)
    disp_dl  = main('templ.png','tempr.png')
    disp, _ = stereoMatchSGBM(iml_rectified_l, imr_rectified_r, down_scale= False, up_scale=False) 
    cv2.imwrite('dispart.png',np.concatenate((disp,disp_dl),axis=0).astype(np.uint8))
    # 计算像素点的3D坐标（左相机坐标系下）
    points_3d = cv2.reprojectImageTo3D(disp, Q)  # 可以使用上文的stereo_config.py给出的参数
    points_3d_dl = cv2.reprojectImageTo3D(disp_dl, Q)
    global last_point
    last_point = [0,0,0] # 记录上一次点击的点（用于和下个点计算）
    global last_point_dl
    last_point_dl = [0,0,0] # 记录上一次点击的点（用于和下个点计算）
    # 鼠标点击事件
    def onMouse(event, x, y, flags, param):
        global last_point
        global last_point_dl
        if event == cv2.EVENT_LBUTTONDOWN:

            print('点 (%d, %d) 的三维坐标 (%f, %f, %f)' % (x, y, points_3d[y, x, 0], points_3d[y, x, 1], points_3d[y, x, 2]))
            print('点 (%d, %d) 的三维坐标 (%f, %f, %f)' % (x, y, points_3d_dl[y, x, 0], points_3d_dl[y, x, 1], points_3d_dl[y, x, 2]))

            dis = ( (points_3d[y, x, 0] ** 2 + points_3d[y, x, 1] ** 2 + points_3d[y, x, 2] **2) ** 0.5) / 1000
            dis_dl = ( (points_3d_dl[y, x, 0] ** 2 + points_3d_dl[y, x, 1] ** 2 + points_3d_dl[y, x, 2] **2) ** 0.5) / 1000
            print('点 (%d, %d) 距离左摄像头的相对距离为:(1) %0.3f m; (2) %0.3f m ' %(x, y, dis, dis_dl) )
        if event == cv2.EVENT_LBUTTONDOWN:

            print_function(dis, last_point, points_3d, x, y)
            print_function(dis_dl, last_point_dl, points_3d_dl, x, y )
            # 更新点
            last_point_dl = [points_3d_dl[y, x, 0], points_3d_dl[y, x, 1], points_3d_dl[y, x, 2]]
            last_point = [points_3d[y, x, 0], points_3d[y, x, 1], points_3d[y, x, 2]]
        # 显示图片
    # image=cv2.imread('templ.png',2) 
    image1=cv2.applyColorMap(disp_dl.astype(np.uint8), cv2.COLORMAP_JET) * 0.4 + 0.6 * cv2.imread('templ.png') 

    disp_show = iml_rectified_l
    cv2.namedWindow("disparity",0)
    cv2.imshow("disparity", image1.astype(np.uint8))
    cv2.setMouseCallback("disparity", onMouse, 0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
