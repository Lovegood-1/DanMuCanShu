import numpy as np


####################仅仅是一个示例###################################


# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        self.cam_matrix_left = np.array([[1.428e+3, 0, 377.1307],
                                        [0, 1.430e+3, 241.06],
                                                               [0., 0., 1.]])        
        self.cam_matrix_right = np.array([[1.4305e+3, 0, 361.117],
                                         [0, 1.4306e+3, 242.16],
                                                              [0., 0., 1.]])


        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[0.0431, 0.1126, 0,0]])
        self.distortion_r = np.array([[0.0459, -0.1119, 0.00 ,  0.00 ]])

        # 旋转矩阵
        self.R =  np.array([[0.9999, -0.0099, -0.01],
                            [0.0101, 0.9998, 0.0195],
                            [0.0098, -00.0195, 0.998]]).T

        # 平移矩阵
        self.T = np.array([[-128.66], [-0.1829], [-0.3646]])

        # 焦距
        self.focal_length = 1544.5734018116168 # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3] # TODO

        # 基线距离
        self.baseline = 700.1814 # 单位：mm， 为平移向量的第一个参数（取绝对值）

       


