import numpy as np


####################仅仅是一个示例###################################


# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        self.cam_matrix_left = np.array([[1.362e+3, 0, 334.1211],
                                        [0, 1.36170e+3, 200],
                                                               [0., 0., 1.]])        
        self.cam_matrix_right = np.array([[1.3636e+3, 0, 364],
                                         [0, 1.3622e+3, 228],
                                                              [0., 0., 1.]])


        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[-0.1075, 0.7494, 0,0.0, 0]])
        self.distortion_r = np.array([[0.0637, -0.6707, -0.00 ,  0.000,0 ]])

        # 旋转矩阵
        self.R =  np.array([[0.9999, -0.0088, 0.0192],
                            [0.0083, 0.9997, 0.0276],
                            [-0.0194, -00.0274, 0.9998]]).T

        # 平移矩阵
        self.T = np.array([[-255.245], [0.7176], [0.7270]])

        # 焦距
        self.focal_length = 1544.5734018116168 # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3] # TODO

        # 基线距离
        self.baseline = 700.1814 # 单位：mm， 为平移向量的第一个参数（取绝对值）

       


