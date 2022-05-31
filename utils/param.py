import numpy as np


####################仅仅是一个示例###################################


# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        self.cam_matrix_left = np.array([[1.356e+3, 0, 386.657],
                                        [0, 1.3544e+3, 201.16],
                                                               [0., 0., 1.]])        
        self.cam_matrix_right = np.array([[1.3565e+3, 0, 390.26],
                                         [0, 1.3554e+3, 210.56],
                                                              [0., 0., 1.]])


        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[0.02, 0.3, 0,0]])
        self.distortion_r = np.array([[0.0106, 0.1381, 0.00 ,  0.00 ]])

        # 旋转矩阵
        self.R =  np.array([[0.9999, -0.0107, 0],
                            [0.0107, 1, 0.0195],
                            [0, -00.0195, 1]]).T

        # 平移矩阵
        self.T = np.array([[-122.0493], [-0.3556], [-0.5373]])

        # 焦距
        self.focal_length = 1544.5734018116168 # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3] # TODO

        # 基线距离
        self.baseline = 700.1814 # 单位：mm， 为平移向量的第一个参数（取绝对值）

       


