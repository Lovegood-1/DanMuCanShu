from detect_api2 import detect2
from optflow_diff_api import frame_diff, video_process
# from twoeye import rebuild
import shutil
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import time
# from twoeye import rebuild
from utils.torch_utils import get_max_bbox, show_video
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import webbrowser
from utils.to_word_cord import camera2word
def video_preprocess(path, out1, out2):
    # 将双目相机拍摄的视频裁切开，处理完之后进行保存

    cap = cv2.VideoCapture(path)

    ret, frame = cap.read()
    h,w,c = frame.shape

    fps=60
    fourcc = cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')
    videoWriter1 = cv2.VideoWriter(out1, fourcc, fps, (int(frame.shape[1]/2), frame.shape[0]), True)
    videoWriter2 = cv2.VideoWriter(out2, fourcc, fps, (int(frame.shape[1]/2), frame.shape[0]), True)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame1, frame2 = frame[:,:int(w/2),:], frame[:,int(w/2):,:]
        videoWriter1.write(frame1)
        videoWriter2.write(frame2)

    cap.release()
    videoWriter1.release()
    videoWriter2.release()


def localization2(path, rebuild):
    """_summary_

    Args:
        path (_type_): 一次性输入两个视频的路径

    Returns:
        _type_: _description_
        fire_left: the pixel coordinates of bbox left bottom in both camera
            (4,) for a bbox:[ul,vl,ur,vr]
        fire_right: the pixel coordinates of bbox right bottom in both camera
            (4,) for a bbox:[ul,vl,ur,vr]
        pixel_distance_of_high: the pixel distance of the height of bbox in left camera
            (1,) for a bbox  
    """
    dict_, dict_2 = detect2(path) # 返回 两个视频的字典
    dict1, dict_all1 = fire_info(dict_) # 第一帧的索引
    dict2, dict_all2 = fire_info(dict_2)
    # HW = 0
    fire_left, fire_right, pixel_distance_of_high ,left_bbox, fire_frame=  get_max_bbox(path, dict_, dict_2)
    d = rebuild.triangulation_double_bbox(fire_left, fire_right)
    h = (pixel_distance_of_high/(fire_left[2] - fire_left[1])) * d

    return dict1, dict_all1, dict2, dict_all2, d, h, left_bbox, fire_frame


def fire_info(dict_):
    for key, value in dict_.items():
        # print(key.type)
        for v in value:
            v_ = v.split(' ')
            isfire = v_[0]
            if isfire == '0':
                center_x = float(v_[1])
                center_y = float(v_[2])
                w = float(v_[3])
                h = float(v_[4])
                bl_x = center_x-w/2
                bl_y = center_y+h/2
                br_x = center_x+w/2
                br_y = center_y+h/2
                # 返回 存在火焰帧的第一帧的索引， 火焰边框坐标（左上角，右下角），dict_：火焰检测结果（帧索引，框的中心点坐标。框的宽和高）
                fire_info = [int(key), bl_x, bl_y, br_x, br_y]
                return [int(key), bl_x, bl_y, br_x, br_y], dict_
    return


def WriteText(dd_x, dd_y, dd_z, radium_list, height_list, speed, angle_qing, angle_pian, angle_pian_direction):
    # dd_z=0

    print(angle_pian_direction)
    angle_pian_direction = angle_pian_direction.replace('前','南')
    angle_pian_direction = angle_pian_direction.replace('后','北')
    angle_pian_direction = angle_pian_direction.replace('左','东')
    angle_pian_direction = angle_pian_direction.replace('右','西')

    # 保存检测结果至txt文件
    print(angle_qing)

    locating_x_str = str(np.around(dd_x,2)) + '\n'
    locating_y_str = str(np.around(dd_y,2)) + '\n'
    locating_z_str = str(np.around(dd_z,2)) + '\n'

    speed_str = str(np.around(speed[0],2)) + '\n'

    angle_qing_str = str(np.around(angle_qing[0],2)) + '\n'

    angle_pian_str = angle_pian_direction + str(np.around(angle_pian[0],2)) + '\n'

    
    if len(radium_list) == 0 : # BUG
        radium_list = [0.5]
        
    fire_radium_max_str = str(np.around(max(radium_list),2)) + '\n'

    if len(height_list) == 0 :
        height_list = [0.5]
    fire_height_max_str = str(np.around(max(height_list),2)) + '\n'

    danmu_txt = 'D:/view32/view32/data/' +'data.txt'
    danmu = open(danmu_txt, 'w', encoding='utf-8')
    danmu.write(locating_x_str)
    danmu.write(locating_y_str)
    danmu.write(locating_z_str)
    danmu.write(speed_str)
    danmu.write(angle_qing_str)
    danmu.write(angle_pian_str)
    danmu.write(fire_radium_max_str)
    danmu.write(fire_height_max_str)
    danmu.close()

    # weili_txt = root+'weili.txt'
    # weili = open(weili_txt, 'w', encoding='utf-8')
    # weili.write(angle_qing_str)
    # weili.write(angle_pian_str)
    # weili.write(locating_str)
    # weili.write(locating_coordinates)
    # weili.write(speed_str)
    # weili.write(speed_number)
    # weili.write(fire_radium_max_str)
    # weili.write(fire_radium_max)
    # weili.write(fire_height_max_str)
    # weili.write(fire_height_max)
    # weili.close()

def compute_FireSize_DDSpeed_Angle(fire_range1, fire_range2, 
                                    realstart_x, realstart_y, realstart_z,
                                    realend_x, realend_y, realend_z,
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
    L = np.sqrt((y_diff)*(y_diff)+horizotal*horizotal)
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



if __name__=='__main__':


    root = 'video\\' 

    # # # 直播检测，保存视频 # 当前版本不用
    # from detect_OTA import detect  
    # detect()

    # # 矫正视频，输出拼接的视频
    import  calibration.param_0531 as stereoconfig
    from utils.video_rectify import video_rectify_double
    left_video = os.path.join(root,'camera_rgb_double.avi')
    config = stereoconfig.stereoCamera()
    Video = video_rectify_double(left_video,  config)
    Video.rectify_video_double()
    rebuild = Video.triangulation

    
    path = root+'camera_rgb_double_rectify_all_.avi'
    out1 = root+'camera_leftrectify.avi'
    out2 = root+'camera_rightrectify.avi'

    print('▅'*80,'Step1： Video Preprocess')
    video_preprocess(path, out1, out2)

    print('▅'*80,'Step2： Finding Fire by YOLO')
    dict1, dict_all1, dict2, dict_all2, W, H, left_bbox, fire_frame = localization2(out1 + ',' + out2, Video)

    print('▅'*80,'Step3： Tracking Object for Video1')
    track_index1, track_location1, result_path1, fire_range1, x_fire1, y_fire1 = \
                                                    video_process(root, out1, dict1, dict_all1, W = W, H = H,  left_bbox = left_bbox, fire_frame= fire_frame)

    print('▅'*80,'Step4： Tracking Object for Video2')
    track_index2, track_location2, result_path2, fire_range2, x_fire2, y_fire2 = \
                                                    video_process(root, out2, dict2, dict_all2, left_bbox)

    print('▅'*80,'Step5： Smoothing Tracked Points')
    startpoint1, endpoint1, startpoint2, endpoint2 = -1,-1,-1,-1
    startindex, endindex = -1,-1

    for i1, l1 in zip(track_index1, track_location1):
        index = np.where(track_index2==i1)
        if index[0].shape[0]==1:
            startindex = i1
            startpoint1, startpoint2 = l1, track_location2[index[0][0]]
            break
        if index[0].shape[0]>1:
            sub_points = track_location2[index[0]]
            dis_list = np.abs(sub_points[:,0]-l1[0])+np.abs(sub_points[:,1]-l1[1])
            min_index = np.argmin(dis_list)

            startindex = i1
            startpoint1, startpoint2 = l1, track_location2[index[0][min_index]]
            break

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
    
    # t_index1, t_location1, ymin = [],[],0
    # for i, l in zip(track_index1,track_location1):
    #     if l[1]>ymin: ymin=l[1]; t_index1.append(i); t_location1.append(l);

    # t_index2, t_location2, ymin = [],[],0
    # for i, l in zip(track_index2,track_location2):
    #     if l[1]>ymin: ymin=l[1]; t_index2.append(i); t_location2.append(l);
    time.sleep(0.5)
    # show_video(out1,window_name=window_name) # 接着显示处理后的视频
    # show_video(out2, window_name)
    print('▅'*80,'Step5： Rebuilding Coordinates')
    fire_x, fire_y, fire_z = rebuild(x_fire1, y_fire1, x_fire2, y_fire2)
    realstart_x, realstart_y, realstart_z = rebuild(startpoint1[0], startpoint1[1], startpoint2[0], startpoint2[1])
    realend_x, realend_y, realend_z = rebuild(endpoint1[0], endpoint1[1], endpoint2[0], endpoint2[1])
    


    print('▅'*80,'Step6： Rebuilding Fire Size')
    radium_list, height_list, speed, angle_qing, angle_pian, angle_pian_direction = \
                        compute_FireSize_DDSpeed_Angle(fire_range1, fire_range2, 
                                                        realstart_x, realstart_y, realstart_z,
                                                        realend_x, realend_y, realend_z,
                                                        startindex, endindex)

    # print('▅'*80,'Step7： Saving txt information')
    # Add Text
    fire_x, fire_y, fire_z = camera2word(fire_x, fire_y, fire_z)
    radium_list, height_list = [W/2], [H]
    WriteText(fire_x[0], fire_y[0], fire_z[0], radium_list, height_list, 
                    speed, angle_qing, angle_pian, angle_pian_direction)
    print('{} processed !'.format(result_path1))
    WriteText(fire_x[0], fire_y[0], fire_z[0], radium_list, height_list, 
                    speed, angle_qing, angle_pian, angle_pian_direction)
    print('{} processed !'.format(result_path2))
    # 自动显示后处理结果

    time.sleep(1)
    web_path = 'http://localhost:8088/danmu.html'
    print('打开网页', web_path)
    webbrowser.open(web_path)
