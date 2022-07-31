"""2022年7月30日09:55:23
测试利用双目确定世界坐标  https://www.jianshu.com/p/77f7c0cd9ec7 
"""
#!/usr/bin/env python3
# coding:utf-8
import os
import re
import cv2
from matplotlib import image
import numpy as np
import time
from PIL import Image,ImageTk
import math
import csv
import pandas as pd

class PNPSolver(): 
    def __init__(self, Intrinsic, waican , size_wh = (4.96, 3.72), imageWH = (1920,1080), img_path = r'', camera_id = '192.168.0.2'):
        self.COLOR_WHITE = (255,255,255)
        self.COLOR_BLUE = (255,0,0)
        self.COLOR_LBLUE = (255, 200, 100)
        self.COLOR_GREEN = (0,240,0)
        self.COLOR_RED = (0,0,255)
        self.COLOR_DRED = (0,0,139)
        self.COLOR_YELLOW = (29,227,245)
        self.COLOR_PURPLE = (224,27,217)
        self.COLOR_GRAY = (127,127,127)   
        self.Points3D = np.zeros((1, 4, 3), np.float32)  #存放4组世界坐标位置 TODO: 自适应数量，不一定是4
        self.Points2D = np.zeros((1, 4, 2), np.float32)   #存放4组像素坐标位置 TODO: 自适应数量，不一定是4
        self.point2find = np.zeros((1, 2), np.float32)
        self.cameraMatrix = None
        self.distCoefs = None
        self.f = 0
        self.img_path = img_path
        self.size_wh = size_wh
        self.imageWH = imageWH
        self.camera_id = camera_id
        self.filename = Intrinsic
        self.waican = waican 
        self.init_Point()

    def rotationVectorToEulerAngles(self, rvecs, anglestype):
        R = np.zeros((3, 3), dtype=np.float64)
        cv2.Rodrigues(rvecs, R)
        sy = math.sqrt(R[2,1] * R[2,1] +  R[2,2] * R[2,2])
        singular = sy < 1e-6
        if  not singular:
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        if anglestype == 0:
            x = x*180.0/3.141592653589793
            y = y*180.0/3.141592653589793
            z = z*180.0/3.141592653589793
        elif anglestype == 1:
            x = x
            y = y
            z = z
        print(x)
        return x,y,z

    def CodeRotateByZ(self, x,  y,  thetaz):#将空间点绕Z轴旋转
        x1=x #将变量拷贝一次，保证&x == &outx这种情况下也能计算正确
        y1=y
        rz = thetaz*3.141592653589793/180
        outx = math.cos(rz)*x1 - math.sin(rz)*y1
        outy = math.sin(rz)*x1 + math.cos(rz)*y1
        return outx,outy

    def CodeRotateByY(self, x, z, thetay):
        x1=x
        z1=z
        ry = thetay * 3.141592653589793 / 180
        outx = math.cos(ry) * x1 + math.sin(ry) * z1
        outz = math.cos(ry) * z1 - math.sin(ry) * x1
        return outx,outz

    def CodeRotateByX(self, y, z, thetax):
        y1=y
        z1=z
        rx = (thetax * 3.141592653589793) / 180
        outy = math.cos(rx) * y1 - math.sin(rx) * z1
        outz = math.cos(rx) * z1 + math.sin(rx) * y1
        return outy,outz

    def solver(self):
        retval, self.rvec, self.tvec = cv2.solvePnP(self.Points3D, self.Points2D, self.cameraMatrix, self.distCoefs)
        thetax,thetay,thetaz = self.rotationVectorToEulerAngles(self.rvec, 0)
        x = self.tvec[0][0]
        y = self.tvec[1][0]
        z = self.tvec[2][0]
        self.Position_OwInCx = x
        self.Position_OwInCy = y
        self.Position_OwInCz = z
        self.Position_theta = [thetax, thetay, thetaz]
        #print('Position_theta:',self.Position_theta)
        x, y = self.CodeRotateByZ(x, y, -1 * thetaz)
        x, z = self.CodeRotateByY(x, z, -1 * thetay)
        y, z = self.CodeRotateByX(y, z, -1 * thetax)
        self.Theta_W2C = ([-1*thetax, -1*thetay,-1*thetaz])
        self.Position_OcInWx = x*(-1)
        self.Position_OcInWy = y*(-1)
        self.Position_OcInWz = z*(-1)
        self.Position_OcInW = np.array([self.Position_OcInWx, self.Position_OcInWy, self.Position_OcInWz])
        print('Position_OcInW:', self.Position_OcInW)

    def WordFrame2ImageFrame(self, WorldPoints):
        pro_points, jacobian = cv2.projectPoints(WorldPoints, self.rvecs, self.tvecs, self.cameraMatrix, self.distCoefs)
        return pro_points

    def ImageFrame2CameraFrame(self, pixPoints):
        fx = self.cameraMatrix[0][0]
        u0 = self.cameraMatrix[0][2]
        fy = self.cameraMatrix[1][1]
        v0 = self.cameraMatrix[1][2]
        zc = (self.f[0]+self.f[1])/2
        xc = (pixPoints[0] - u0) * self.f[0] / fx  #f=fx*传感器尺寸/分辨率
        yc = (pixPoints[1] - v0) * self.f[1] / fy
        point = np.array([xc,yc,zc])
        return point

    def getudistmap2(self, filename):
        with open(filename, 'r',newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            rows = [row for row in spamreader]
            self.cameraMatrix = np.zeros((3, 3))
            #Dt = np.zeros((4, 1))
            size_w = 4.96 # TODO
            size_h = 3.72

            imageWidth = 1920
            imageHeight = 1080
            self.cameraMatrix[0][0] = rows[1][1]
            self.cameraMatrix[1][1] = rows[1][2]
            self.cameraMatrix[0][2] = rows[1][3]
            self.cameraMatrix[1][2] = rows[1][4]
            self.cameraMatrix[2][2] = 1
            print(len(rows[2]))
            if len(rows[2]) == 5:
                print('fisheye')
                self.distCoefs = np.zeros((4, 1))
                self.distCoefs[0][0] = rows[2][1]
                self.distCoefs[1][0] = rows[2][2]
                self.distCoefs[2][0] = rows[2][3]
                self.distCoefs[3][0] = rows[2][4]
                scaled_K = self.cameraMatrix * 0.8 # The values of K is to scale with image dimension.
                scaled_K[2][2] = 1.0                   
            else:
                print('normal')
                self.distCoefs = np.zeros((1, 5))
                self.distCoefs[0][0] = rows[2][1]
                self.distCoefs[0][1] = rows[2][2]
                self.distCoefs[0][2] = rows[2][3]
                self.distCoefs[0][3] = rows[2][4]
                self.distCoefs[0][4] = rows[2][5]     
            print('dim = %d*%d'%(imageWidth, imageHeight))
            print('Kt = \n', self.cameraMatrix)
            print('Dt = \n', self.distCoefs)
            self.f = [self.cameraMatrix[0][0]*(size_w/imageWidth), self.cameraMatrix[1][1]*(size_h/imageHeight)]
            print('f = \n', self.f)
            return 0

    def getudistmap(self, filename = r'calibration\cameraParams_new0727.mat'):
        from calibration.load_cameraParams import load_mat
        # spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        params = load_mat(self.filename)
        # rows = [row for row in spamreader]
        self.cameraMatrix = np.zeros((3, 3))
        try:
            size_w = self.size_wh[0]
            size_h = self.size_wh[-1]
            imageWidth = self.imageWH[0]
            imageHeight = self.imageWH[-1]
        except:
            size_w = 4.96  
            size_h = 3.72
            imageWidth = 1920
            imageHeight = 1080

        self.cameraMatrix[0][0] = params['IntrinsicMatrix'][0][0][0,0]
        self.cameraMatrix[1][1] = params['IntrinsicMatrix'][0][0][1,1]
        self.cameraMatrix[0][2] = params['IntrinsicMatrix'][0][0][2,0]
        self.cameraMatrix[1][2] = params['IntrinsicMatrix'][0][0][2,1]
        self.cameraMatrix[2][2] = 1
        self.distCoefs = np.zeros((1, 5))
        self.distCoefs[0][0] = params['RadialDistortion'][0][0][0][0]
        self.distCoefs[0][1] = params['RadialDistortion'][0][0][0][1]
        self.distCoefs[0][2] = params['TangentialDistortion'][0][0][0][0]
        self.distCoefs[0][3] = params['TangentialDistortion'][0][0][0][0]
        self.distCoefs[0][4] = 0  
        print('dim = %d*%d'%(imageWidth, imageHeight))
        print('Kt = \n', self.cameraMatrix)
        print('Dt = \n', self.distCoefs)
        self.f = [self.cameraMatrix[0][0]*(size_w/imageWidth), self.cameraMatrix[1][1]*(size_h/imageHeight)]
        print('f = \n', self.f)
        return 0

    def show_2dpoints(self, img = None, final_3dpoint = -1):
        if  img is None:
            img = cv2.imread(self.img_path)
            for num_ in range(self.Points2D.shape[1]): # 画原来的点
                cv2.circle(img, (int(self.Points2D[0,num_,0]), int(self.Points2D[0,num_,1])), 3, (0,0,255), -1)
                nose_end_point, jacobian = cv2.projectPoints(self.Points3D[0,num_], self.rvec, self.tvec,  self.cameraMatrix, self.distCoefs)
                cv2.circle(img, (int(nose_end_point[0,0,0]), int(nose_end_point[0,0,1])), 3, (255,255,255), -1)
            
            return img
        else:
            cv2.circle(img, (int(self.point2find[0]), int(self.point2find[1])), 3, (0 , 255 , 0), -1) # 要找的点（像素）
            nose_end_point, jacobian = cv2.projectPoints(final_3dpoint, self.rvec, self.tvec,  self.cameraMatrix, self.distCoefs) # 要找的点（真实）
            cv2.circle(img, (int(nose_end_point[0,0,0]), int(nose_end_point[0,0,1])), 3, (100 , 255 , 100), -1)
            return img

    def init_Point(self):
        """读取外参文件来为 2d 3d 点赋值

        Returns:
            _type_: _description_
        """
        assert os.path.isfile(self.waican)
        data = pd.read_csv(self.waican)
        pixel_cor1 = data[['u1','v1']] # 列名称
        pixel_cor1 = np.array(pixel_cor1).reshape(1,-1,2)
        self.Points2D = pixel_cor1.astype(np.float32)
        worldpoint1 = data[['x1','y1','z1']]
        worldpoint1 = np.array(worldpoint1,dtype=np.float32)
        self.Points3D = np.expand_dims(worldpoint1,0)
        self.getudistmap()
        self.solver()

        # 2 外参映射完成


class GetDistanceOf2linesIn3D():
    def __init__(self):
        print('GetDistanceOf2linesIn3D class')

    def dot(self, ax, ay, az, bx, by, bz):
        result = ax*bx + ay*by + az*bz
        return result

    def cross(self, ax, ay, az, bx, by, bz):
        x = ay*bz - az*by
        y = az*bx - ax*bz
        z = ax*by - ay*bx
        return x,y,z

    def crossarray(self, a, b):
        x = a[1]*b[2] - a[2]*b[1]
        y = a[2]*b[0] - a[0]*b[2]
        z = a[0]*b[1] - a[1]*b[0]
        return np.array([x,y,z])

    def norm(self, ax, ay, az):
        return math.sqrt(self.dot(ax, ay, az, ax, ay, az))

    def norm2(self, one):
        return math.sqrt(np.dot(one, one))


    def SetLineA(self, A1x, A1y, A1z, A2x, A2y, A2z):
        self.a1 = np.array([A1x, A1y, A1z]) 
        self.a2 = np.array([A2x, A2y, A2z])

    def SetLineB(self, B1x, B1y, B1z, B2x, B2y, B2z):
        self.b1 = np.array([B1x, B1y, B1z])    
        self.b2 = np.array([B2x, B2y, B2z])

    def GetDistance(self):
        d1 = self.a2 - self.a1
        d2 = self.b2 - self.b1
        e = self.b1 - self.a1

        cross_e_d2 = self.crossarray(e,d2)
        cross_e_d1 = self.crossarray(e,d1)
        cross_d1_d2 = self.crossarray(d1,d2)

        dd = self.norm2(cross_d1_d2)
        t1 = np.dot(cross_e_d2, cross_d1_d2)
        t2 = np.dot(cross_e_d1, cross_d1_d2)

        t1 = t1/(dd*dd)
        t2 = t2/(dd*dd)

        self.PonA = self.a1 + (self.a2 - self.a1) * t1
        self.PonB = self.b1 + (self.b2 - self.b1) * t2

        self.distance = self.norm2(self.PonB - self.PonA)
        print('distance=', self.distance)
        return self.distance

def main_test_pnp():
    calibration_dir
    camera_info = {}
    camera_list = os.listdir(calibration_dir) # 获取所有 mat 文件
    for camera in camera_list:
        ip_ = re.findall( r'[0-9]+(?:\.[0-9]+){3}',camera)[0]
        if len(ip_) > 0:
            camera_info[ip_] = PNPSolver(os.path.join(calibration_dir,camera))

    pass

if __name__== '__main__':
    main_test_pnp()