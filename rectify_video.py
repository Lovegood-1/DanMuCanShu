"""
单张图片矫正步骤
1.1 # 读取相机内参和外参
1.2 # 立体校正
1.3 # 消除畸变
"""
from cmath import rect
import cv2 
import os.path as osp
from cv2 import CALIB_ZERO_DISPARITY
import numpy as np
from torch import det
class video_rectify:
    def __init__(self, left, right, stereoconfig) -> None:
        from cv2 import CALIB_ZERO_DISPARITY
        # check the fps and hw of the video
        self.info_dict = self.get_video_info(left) 
        assert  self.info_dict == self.get_video_info(right)

        # 1.1
        self.config = stereoconfig
        # 1.2
        self.getRectifyTransform()
        # # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵


    # 畸变校正和立体校正
    def rectify_single_image(self, img, left_or_right):
        if left_or_right == 'left':
            rectifyed_img1 = cv2.remap(img, self.map1x, self.map1y, cv2.INTER_AREA)
        elif left_or_right == 'right':
            rectifyed_img1 = cv2.remap(img, self.map2x, self.map2y, cv2.INTER_AREA)
        else:
            assert ValueError
        return rectifyed_img1

    def rectify_video(self, video_path_l, left_or_right):
        # 读取视频
        assert osp.isfile(video_path_l) == True
        cap = cv2.VideoCapture(video_path_l)
        save_p = video_path_l.split('.')[0] + 'rectify' + '.' + video_path_l.split('.')[-1]
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)

        print('w: {}, h: {}, count: {}, fps: {}'.format(w, h, count, fps))

        # 视频保存
        # fourcc = cv2.VideoWriter_fourcc('P', 'I', 'M', '1')
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # 视频编码格式
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print('save to >>>',save_p)
        out = cv2.VideoWriter(save_p, fourcc, fps, (int(w), int(h)), True)
        # 获取矫正后的图片
        while cap.isOpened():
            ret, frame = cap.read()
            # 调用本地摄像头时，需要左右翻转一下，若是视频文件则不需要翻转
            # frame = cv2.flip(frame, 1)
            if not ret:
                break
            rec_frame = self.rectify_single_image(frame, left_or_right)
            cv2.imshow('a',np.concatenate((frame, rec_frame), axis = 1))
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration
            out.write(rec_frame)
            # # 保存
        out.release()
        pass

    def getRectifyTransform(self):
        # 读取内参和外参
        left_K = self.config.cam_matrix_left
        right_K = self.config.cam_matrix_right
        left_distortion = self.config.distortion_l
        right_distortion = self.config.distortion_r
        R = self.config.R
        T = self.config.T
        # R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion, 
        #                                             (self.info_dict['W'], self.info_dict['H']), R, T, alpha=1, flags=CALIB_ZERO_DISPARITY)
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion, 
                                                (self.info_dict['W'], self.info_dict['H']), R, T, alpha=0) # 带有黑边
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (self.info_dict['W'], self.info_dict['H']), cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (self.info_dict['W'], self.info_dict['H']), cv2.CV_32FC1)
            
    def get_video_info(self, video_path):

        assert osp.isfile(video_path) == True
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        info_dict = {
            'W': w,
            'H': h,
            'count': count,
            'fps': fps,
        }
        cap.release()
        return info_dict

if __name__ == '__main__':
    import  utils.param as stereoconfig_040_2   #导入相机标定的参数

    # 1 build obj  # TODO: better pack to a pair + a config
    left_video = r'video\left.mp4'
    right_video = r'video\right.mp4'
    config = stereoconfig_040_2.stereoCamera()
    Video = video_rectify(left_video, right_video, config)
    Video.rectify_video(left_video, 'left')
    Video.rectify_video(right_video, 'right')
    from detect_from_camera import detect
    imgl, imgr, bbox_l = detect()
    imgl, imgr = Video.rectify_single_image(imgl,'left'), Video.rectify_single_image(imgl,'right')
    

    # 读取相机内参和外参

    # 对于某个照片
        # 立体矫正
        # 消除畸变
        # 保存