"""
建立一个用于矫正保存视频的对象
"""

from typing import Tuple
import cv2 
import os.path as osp
import math
import numpy as np


class video_rectify:
    def __init__(self, left, right, stereoconfig) -> None:
        from cv2 import CALIB_ZERO_DISPARITY
        # check the fps and hw of the video
        self.left_path = left
        self.right_path = right
        self.info_dict = self.get_video_info(left) 
        assert  self.info_dict == self.get_video_info(right)

        # get the intrin 
        self.config = stereoconfig
        self.getRectifyTransform()
        # # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵



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
    def rectify_video_double(self):
        # 读取视频
        assert osp.isfile(self.left_path) == True
        cap_l = cv2.VideoCapture(self.left_path)
        save_p = self.left_path.split('.')[0] + 'rectify_all_' + '.' + self.left_path.split('.')[-1]
        w = cap_l.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT)
        count = cap_l.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap_l.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # out_l = cv2.VideoWriter(save_p, fourcc, fps, (int(w), int(h)), True)

        assert osp.isfile(self.right_path) == True
        cap_r = cv2.VideoCapture(self.right_path)
        # save_p = video_path_r.split('.')[0] + 'rectify_all_' + '.' + video_path_r.split('.')[-1]
        # w = cap_r.get(cv2.CAP_PROP_FRAME_WIDTH)
        # h = cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # count = cap_r.get(cv2.CAP_PROP_FRAME_COUNT)
        # fps = cap_r.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print('save to >>>',save_p)
        # out_r = cv2.VideoWriter(save_p, fourcc, fps, (int(w), int(h)), True)
        out = cv2.VideoWriter(save_p, fourcc, fps, (int(w) * 2, int(h)))


        # 获取矫正后的图片
        while cap_l.isOpened() and cap_r.isOpened():
            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()
            # 调用本地摄像头时，需要左右翻转一下，若是视频文件则不需要翻转
            # frame = cv2.flip(frame, 1)
            if not ret_l:
                break
            rec_frame_l = self.rectify_single_image(frame_l, 'left')
            rec_frame_r = self.rectify_single_image(frame_r, 'right')
            cv2.imshow('a',np.concatenate((frame_l, frame_r), axis = 1))
            cv2.imshow('rec',np.concatenate((rec_frame_l, rec_frame_r), axis = 1))
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration
            out.write(np.concatenate((rec_frame_l, rec_frame_r), axis = 1))
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
        R1, R2, P1, P2, self.Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion, 
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


class video_rectify_double:
    def __init__(self, left , stereoconfig) -> None:
        from cv2 import CALIB_ZERO_DISPARITY
        # check the fps and hw of the video
        self.left_path = left
        self.info_dict = self.get_video_info(left) 

        # get the intrin 
        self.config = stereoconfig
        self.getRectifyTransform()
        # # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵



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
    def rectify_video_double(self):
        """矫正拼接视频，并输出拼接视频

        Raises:
            StopIteration: _description_
        """
        # 读取视频
        assert osp.isfile(self.left_path) == True
        cap_l = cv2.VideoCapture(self.left_path)
        save_p = self.left_path.split('.')[0] + '_rectify_all_' + '.' + self.left_path.split('.')[-1]
        w = cap_l.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT)
        count = cap_l.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap_l.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_p, fourcc, fps, (int(w), int(h)))
        print('Fun rectify_video_double save >>>', save_p )

        # 获取矫正后的图片
        while cap_l.isOpened():
            ret_l, frame = cap_l.read()
            if ret_l is True:
                frame_l = frame[:,int(w//2):,...]
                frame_r = frame[:,:int(w//2),...]
                rec_frame_l = self.rectify_single_image(frame_l, 'left')
                rec_frame_r = self.rectify_single_image(frame_r, 'right')

                # cv2.imshow('rec',self.draw_line(rec_frame_l, rec_frame_r))
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
                out.write(np.concatenate((rec_frame_l, rec_frame_r), axis = 1))
                # # 保存
            else:
                break
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
        R1, R2, P1, P2, self.Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion, 
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
            'W': int(w//2),
            'H': h,
            'count': count,
            'fps': fps,
        }
        cap.release()
        return info_dict

    def draw_line(self, image1, image2):
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

    def triangulation(self, x0, y0, x1, y1):
        # disp = np.zeros((self.info_dict['H'], self.info_dict['W']))
        disp = np.abs(x0 - x1)
        if np.abs(y0 - y1) > 10:
            print('由于像素不在一个水平线上面，点的三维重建误差较大')
        vec_tmp = np.array([[x0,y0,disp,1]]).T
        vec_tmp = self.Q @ vec_tmp
        vec_tmp /= vec_tmp[3]
        vec_tmp /= 1000
        return vec_tmp[0], vec_tmp[1], vec_tmp[2]        
    def triangulation_double_bbox(self,pt_1,pt_2):
        pt_lx, pt_ly, pt_rx, pt_ry = pt_1
        pt2_lx, pt2_ly, pt2_rx, pt2_ry = pt_2
        # disp = np.zeros((self.info_dict['H'], self.info_dict['W']))
        # disp = np.abs(x0 - x1)
        disp_bbox = (np.abs(pt_lx - pt_rx) + np.abs(pt2_lx - pt2_rx))/2

 
        vec_tmp = np.array([[pt_lx, pt_ly,disp_bbox,1]]).T
        vec_tmp = self.Q @ vec_tmp
        vec_tmp /= vec_tmp[3]
        vec_tmp /= 1000
        vec_tmp_l = vec_tmp 
        vec_tmp = np.array([[pt2_lx, pt2_ly,disp_bbox,1]]).T
        vec_tmp = self.Q @ vec_tmp
        vec_tmp /= vec_tmp[3]
        vec_tmp /= 1000

        a = self.cal_distance(np.squeeze(np.array(vec_tmp_l)), np.squeeze(np.array(vec_tmp)))

        return a  
    def cal_distance(self, p1,p2,vertical = False):
        return math.sqrt(math.pow(p2[0] - p1[0],2) + math.pow(p2[1] - p1[1],2) + math.pow(p2[2] - p1[2],2))
