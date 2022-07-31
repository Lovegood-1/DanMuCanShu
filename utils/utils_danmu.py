import numpy as np
import time
import math
def fiter_opt_point(track_index1, track_index2, track_location1, track_location2 ):
    startpoint1, endpoint1, startpoint2, endpoint2 = -1,-1,-1,-1
    startindex, endindex = -1,-1
    # 获得导弹的初始点，同时确保帧的序号是对齐的
    for i1, l1 in zip(track_index1, track_location1): # track_index: list, save the indexs of frame where moving objects appear
        index = np.where(track_index2==i1)
        if index[0].shape[0]==1: # 如果有单个点
            startindex = i1
            startpoint1, startpoint2 = l1, track_location2[index[0][0]]
            break
        if index[0].shape[0]>1: # 如果一帧有多个动点
            sub_points = track_location2[index[0]]
            dis_list = np.abs(sub_points[:,0]-l1[0])+np.abs(sub_points[:,1]-l1[1])
            min_index = np.argmin(dis_list)

            startindex = i1
            startpoint1, startpoint2 = l1, track_location2[index[0][min_index]]
            break

        
    # 获得导弹的结束点，同时确保帧的序号是对齐的
    track_index1, track_location1 = track_index1[::-1], track_location1[::-1]
    for i1, l1 in zip(track_index1, track_location1):
        index = np.where(track_index2==i1)
        if index[0].shape[0]==1:
            endindex = i1
            endpoint1, endpoint2 = l1, track_location2[index[0][0]]
            break
        if index[0].shape[0]>1:
            sub_points = track_location2[index[0]]
            dis_list = np.abs(sub_points[:,0]-l1[0])+np.abs(sub_points[:,1]-l1[1])
            min_index = np.argmin(dis_list)

            endindex = i1
            endpoint1, endpoint2 = l1, track_location2[index[0][min_index]]
            break

    if startindex==-1 or endindex==-1 or endindex<=startindex or endpoint1[1]<=startpoint1[1] or endpoint2[1]<=startpoint2[1]:
        startindex, endindex = track_index1[0], track_index1[-1]
        startpoint1, endpoint1, startpoint2, endpoint2 = track_location1[0], track_location1[-1], track_location2[0], track_location2[-1]

    time.sleep(0.5)

    return  startpoint1, endpoint1, startpoint2, endpoint2


def compute_FireSize_DDSpeed_Angle(fire_range1, fire_range2, 
                                    realstart_x, realstart_y,  
                                    realend_x, realend_y, 
                                    startindex, endindex):
    # fire_range : {count : [[tlx, tly, brx, bry], [tlx_r, tly_r, brx_r, bry_r], w, h]}

    # 1. 计算火焰相关参数
    radium_list, height_list = [], []

    dict_keys = fire_range1.keys()
    last_index_ = max(dict_keys)

    for i in range(last_index_+1):
        if fire_range1[i] and fire_range2[i]:
            # 像素坐标
            x11_p, y11_p = fire_range1[i][0][0], fire_range1[i][0][1]
            x12_p, y12_p = fire_range1[i][0][2], fire_range1[i][0][3]

            x21_p, y21_p = fire_range2[i][0][0], fire_range2[i][0][1]
            x22_p, y22_p = fire_range2[i][0][2], fire_range2[i][0][3]

            dist1_p = np.sqrt((x11_p-x12_p)**2+(y11_p-y12_p)**2)
            dist2_p = np.sqrt((x21_p-x22_p)**2+(y21_p-y22_p)**2)

            w_avg_p = 1/2*(dist1_p+dist2_p)

            #真实坐标
            x11_r, y11_r = fire_range1[i][1][0][0]*1e-3, fire_range1[i][1][1][0]*1e-3
            x12_r, y12_r = fire_range1[i][1][2][0]*1e-3, fire_range1[i][1][3][0]*1e-3

            x21_r, y21_r = fire_range2[i][1][0][0]*1e-3, fire_range2[i][1][1][0]*1e-3
            x22_r, y22_r = fire_range2[i][1][2][0]*1e-3, fire_range2[i][1][3][0]*1e-3

            dist1_r = np.sqrt((x11_r-x12_r)**2+(y11_r-y12_r)**2)
            dist2_r = np.sqrt((x21_r-x22_r)**2+(y21_r-y22_r)**2)

            w_avg_r = 1/2*(dist1_r+dist2_r)

            # 计算半径
            radium = w_avg_r/2

            # 计算高度（利用宽度和半径之间的比值关系来直接映射）
            w1, h1, w2, h2 = fire_range1[i][2], fire_range1[i][3], fire_range2[i][2], fire_range2[i][3]

            k = w_avg_r/w_avg_p
            h1_r = h1*k
            h2_r = h2*k
            height = 1/2*(h1_r+h2_r)

            radium_list.append(radium)
            height_list.append(height)

    # 2. 计算DD洛姿
    x_diff, y_diff, z_diff = realend_x-realstart_x, realend_y-realstart_y, realend_z-realstart_z

    # 速度
    time = (endindex-startindex+1)/60
    horizotal = np.sqrt((x_diff)*(x_diff)+(z_diff)*(z_diff))
    L = 60
    speed = L/time

    # 弹道倾角
    angle_qing = np.arctan(np.abs(y_diff)/(horizotal+1e-10))*180/np.pi
    if y_diff<0: angle_qing=-1*angle_qing

    # 弹道偏角
    angle_pian = np.arctan(np.abs(x_diff)/(np.abs(z_diff)+1e-10))*180/np.pi

    # 方向  x轴正负分别对应：西和东， z轴正负分别对应：南和北 
    angle_pian_direction=0
    sin, cos = np.arcsin(z_diff/horizotal)*180/np.pi, np.arccos(x_diff/horizotal)*180/np.pi
    if sin > 0:
        if cos>0: angle_pian_direction = '右偏前'
        elif cos<0: angle_pian_direction = '右偏后'
        else: angle_pian_direction = '正右'
    elif sin==0:
        if cos>0: angle_pian_direction = '正前'
        elif cos<0: angle_pian_direction = '正后'
        else: angle_pian_direction = '垂直向下'
    else:
        if cos>0: angle_pian_direction = '左偏前'
        elif cos<0: angle_pian_direction = '左偏后'
        else: angle_pian_direction = '正左'

    print(realstart_x, realstart_y, realstart_z,
        realend_x, realend_y, realend_z,
        startindex, endindex)
    
    return radium_list, height_list, speed, angle_qing, angle_pian, angle_pian_direction

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
    camera_list = sorted(camera_list ,key = lambda x: ( int(x.split('.')[0]), int(x.split('.')[1]), int(x.split('.')[2]) ))

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