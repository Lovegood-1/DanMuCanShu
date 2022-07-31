


import os
import re
from turtle import towards
import numpy as np
from calibration.calibration_utils import PNPSolver, GetDistanceOf2linesIn3D
from utils.torch_utils import get_file_list

def two_camera(camera_info, p1, p2):
    """利用两个相机的参数计算落点位置

    Args:
        p4psolver1 (_type_): 小ip相机对象
        p4psolver2 (_type_): 大ip相机对象
        p1: 小ip像素坐标
        p2: 大ip像素坐标
        
    """
    # t1_c1 = np.array([1069, 675])
    # t1_c2 = np.array([895, 736])
    camera_list = list(camera_info.keys())
    p4psolver1, p4psolver2 = camera_info[camera_list[0]],camera_info[camera_list[1]]
    point2find1_CF = p4psolver1.ImageFrame2CameraFrame( p1)
    Oc1P_1 = np.array(point2find1_CF)
    Oc1P_1[0], Oc1P_1[1] = p4psolver1.CodeRotateByZ(Oc1P_1[0], Oc1P_1[1], p4psolver1.Theta_W2C[2])
    Oc1P_1[0], Oc1P_1[2] = p4psolver1.CodeRotateByY(Oc1P_1[0], Oc1P_1[2], p4psolver1.Theta_W2C[1])
    Oc1P_1[1], Oc1P_1[2] = p4psolver1.CodeRotateByX(Oc1P_1[1], Oc1P_1[2], p4psolver1.Theta_W2C[0])
    a1 = np.array([p4psolver1.Position_OcInWx, p4psolver1.Position_OcInWy, p4psolver1.Position_OcInWz])
    a2 =  a1 + Oc1P_1
 
    # 4 向量：相机点->未知点 的世界坐标 b2
    point2find2_CF = p4psolver2.ImageFrame2CameraFrame( p2)
    Oc2P_2 = np.array(point2find2_CF)
    Oc2P_2[0], Oc2P_2[1] = p4psolver2.CodeRotateByZ(Oc2P_2[0], Oc2P_2[1], p4psolver2.Theta_W2C[2])
    Oc2P_2[0], Oc2P_2[2] = p4psolver2.CodeRotateByY(Oc2P_2[0], Oc2P_2[2], p4psolver2.Theta_W2C[1])
    Oc2P_2[1], Oc2P_2[2] = p4psolver2.CodeRotateByX(Oc2P_2[1], Oc2P_2[2], p4psolver2.Theta_W2C[0])

    b1 = ([p4psolver2.Position_OcInWx, p4psolver2.Position_OcInWy, p4psolver2.Position_OcInWz])
    b2 = b1 + Oc2P_2

    g = GetDistanceOf2linesIn3D()
    g.SetLineA(a1[0], a1[1], a1[2], a2[0], a2[1], a2[2])
    g.SetLineB(b1[0], b1[1], b1[2], b2[0], b2[1], b2[2])
    distance = g.GetDistance()
    pt = (g.PonA + g.PonB)/2
    return pt

def main_test_pnp():
    calibration_dir = r'calibration\calibration_mat'
    camera_info = {} # {'192.168.1.1':PNPSolver, '192.168.1.2':PNPSolver,}
    camera_list = get_file_list(calibration_dir, ['mat']) # 获取所有 mat 文件
    for camera in camera_list:
        ip_ = re.findall( r'[0-9]+(?:\.[0-9]+){3}',camera)[0]
        if len(ip_) > 0:
            assert os.path.isfile(camera)
            assert  os.path.isfile(os.path.join(calibration_dir,  str(ip_) + '.csv'))
            camera_info[ip_] = PNPSolver(camera, os.path.join(calibration_dir,  str(ip_) + '.csv')) # TODO: 这里有两个文件，以后最好是一个
    # camera_list = list(camera_info.keys())

    two_camera(camera_info  ,  np.array([895, 736]),np.array([1069, 675]))

    pass

if __name__== '__main__':
    main_test_pnp()