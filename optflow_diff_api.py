import numpy as np
import cv2
# from numpy.core.shape_base import block
# import time
# import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw, ImageFont
from twoeye import oneeye


def frame_diff(old, cur):
    threshold_pixel = 0.2

    grey_old = cv2.GaussianBlur(old, (3,3), 0)
    grey_cur = cv2.GaussianBlur(cur, (3,3), 0)
    
    diff = cv2.absdiff(grey_cur, grey_old)

    frame_diff = diff/(np.max(diff)+1e-2)
    frame_diff = frame_diff*frame_diff
    frame_diff[frame_diff>threshold_pixel]=1
    frame_diff = frame_diff*frame_diff

    return frame_diff.astype(np.uint8)*255

def findpoint_fire(L):
    fire_bl_x = L[0]
    fire_bl_y = L[1]
    fire_br_x = L[2]
    fire_br_y = L[3]

    x_fire = (fire_bl_x+fire_br_x)/2
    y_fire = (fire_bl_y+fire_br_y)/2

    return x_fire, y_fire

def findpoint_dd(mask):
    track_point=np.where(mask>0)
    track_point_y = track_point[0]
    track_point_x = track_point[1]
    try:
        y_min_index = np.argmin(track_point_y)

        x_dd = track_point_x[y_min_index]
        y_dd = track_point_y[y_min_index]

    except:
        x_dd = 1
        y_dd = 1

    return x_dd, y_dd


# dict：其中包含了视频所有帧的火焰检测情况（包括未出现火焰的帧信息）
# 当帧中存在火焰，则将其框信息缩放至图像大小，反之则不处理
# 返回的字典里即所有包含火焰的帧索引，以及框信息
def splitdict(dict, shape):
    fire_index_max = 0
    fire_dict = {}
    dictkeys = dict.keys()
    last_index = max(dictkeys)

    for i in range(last_index+1):
        fire_dict.update({i : None})

    for key, value in dict.items():
        tl_x_min = shape[1]
        tl_y_min = shape[1]
        br_x_max = 0
        br_y_max = 0
        # fire_dict.update({key : None})
        have_fire = False
        for v in value:
            v_ = v.split(' ')
            isfire = v_[0]
            if isfire == '0':
                have_fire = True
                center_x = float(v_[1])
                center_y = float(v_[2])
                w = float(v_[3])
                h = float(v_[4])
                tl_x = center_x-w/2
                tl_y = center_y-h/2
                br_x = center_x+w/2
                br_y = center_y+h/2

                if tl_x < tl_x_min:
                    tl_x_min = tl_x
                
                if tl_y < tl_y_min:
                    tl_y_min = tl_y

                if br_x > br_x_max:
                    br_x_max = br_x

                if br_y > br_y_max:
                    br_y_max = br_y

        if have_fire==True:
            if fire_index_max<key:
                fire_index_max=key

            tl_x_min = tl_x_min*shape[1]
            tl_y_min = tl_y_min*shape[0]
            br_x_max = br_x_max*shape[1]
            br_y_max = br_y_max*shape[0]
            w = w*shape[1]
            h = h*shape[0]
            if tl_x_min<br_x_max or tl_y_min<br_y_max:
                fire_dict.update({key : [(int(tl_x_min), int(tl_y_min)),(int(br_x_max), int(br_y_max)), w, h]})

    return fire_dict,fire_index_max



def video_process(root, video_path, dict_first, dict_all, show_fire_point = True):

    # 导弹出现情况统计
    track_index, track_location = [], []

    # 火焰中心点坐标 及 导弹初始点坐标 初始化
    x_fire, y_fire, x_dd, y_dd = 0,0,0,0
    ifdd = True

    # 创建视频读取对象
    cap = cv2.VideoCapture(video_path)

    # 路径定义
    filename = video_path.split('\\')[-1].split('.')[0]
    save_filename = root+'./{}_results.mp4'.format(filename)
    opticalpoints_filename = root+'./{}_opticalpoints.jpg'.format(filename)
    
    # 视频参数设置
    feature_params = dict(maxCorners=100,
                         qualityLevel = 0.01,
                         minDistance = 7,
                         blockSize = 2,
                         useHarrisDetector=True,
                         k = 0.04)
    
    lk_params = dict(winSize = (15,15),
                    maxLevel = 2,
                    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 读取视频首帧
    ret, old_frame = cap.read()

    # 将视频首帧转为灰度图，并作为后续光流追踪的初始化数据
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # p0为初始化的光流追踪点
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    # 后续会将通过光流追踪出来的点可视化到mask上
    mask = (np.zeros_like(old_frame)[:,:,0]).astype(np.uint8)

    # 获取火焰出现的帧索引，以及火焰框信息
    fire_index = dict_first[0]
    fire_bl_x = int(dict_first[1]*old_gray.shape[1])
    fire_bl_y = int(dict_first[2]*old_gray.shape[0])
    fire_br_x = int(dict_first[3]*old_gray.shape[1])
    fire_br_y = int(dict_first[4]*old_gray.shape[0])

    # x_fire,y_fire 为火焰边框中心点坐标，   x_min, y_min 为光流追踪点的最小点坐标
    try:
        x_fire,y_fire = findpoint_fire([fire_bl_x, fire_bl_y, fire_br_x, fire_br_y])
    except:
        pass
    
    # 对 dict_all 进行处理，挑选出拥有火焰的帧，且返回这些帧信息和火焰信息
    # fire_local : [(tlx,tly),(brx,bry),w,h]
    fire_local,fire_index_max= splitdict(dict_all, old_gray.shape)

    # fire_range：即从火焰出现的帧至火焰消失那一帧的帧数范围
    fire_range = {}
    dict_keys = fire_local.keys()
    last_index_ = max(dict_keys)
    for i in range(last_index_+1):
        fire_range.update({i : None})

    # 设置视频存储参数(有些视频帧数可能存在问题，进行一个阈值判断)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print('%'*100)
    # print('fps:',fps)

    # # if is(fps) is 'inf':
    # #     fps=25
    # fps=25
    fps=60

    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    # 创建视频存储对象
    videoWriter = cv2.VideoWriter(save_filename,fourcc,fps,(old_frame.shape[1],old_frame.shape[0]),True)
    count = 0
    max_range = 0

    # 开始逐帧处理，即检测轨迹和火焰
    while True:
        # 视频帧读入，若当前视频为空或已读完就break
        ret, frame = cap.read()
        if not ret: break



        # &&&&&&&&&&  第一阶段：仅视火焰出现【前30帧】为有效帧，对之前的帧进行过滤处理  &&&&&&&&&&
        # fire_index 为火焰出现的帧索引
        if count<fire_index-30:
            print('filtering useless frames-----> fire_index_start:',fire_index, 'fire_index_end:', fire_index_max, 'cur_index:',count)
            count+=1



        # &&&&&&&&&&  第二阶段：对火焰出现【前30帧】至火焰出现【前一帧】的所有帧进行轨迹追踪  &&&&&&&&&&
        elif count<fire_index-1:
            try:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except:
                continue

            # 通过p0光流追踪，若p0光流有问题，则重新在当前帧中初始化追踪点，然后展开追踪
            try:
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            except:
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            good_new = p1[st==1]
            good_old = p0[st==1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                # a:x,  b:y
                a, b = new.ravel()
                c, d = old.ravel()

                #计算每个角点位移距离
                dis = np.sqrt((a-c)**2+(b-d)**2)

                # 计算每个交点的角度,
                #当angle小于0.7时，导弹下落与y轴的夹角小于35度，
                #当angle小于1时，导弹下落与y轴的夹角小于45度，
                #当angle小于1.7时，导弹下落与y轴的夹角小于60度，
                #当angle小于2.8时，导弹下落与y轴的夹角小于70度，
                # angle = np.abs(a-c)/np.abs(b-d)

                # 角点位移>7；  新角点在旧角点下方； 新角点在导弹落点上方； 新旧角点与y轴角度<45度； 
                if (dis>=7) and (b>d) and (b<y_fire):

                    angle_luodian = np.abs(x_fire-c)/np.abs(y_fire-d)
                    # 角点与导弹落点间在y轴角度<35度； 
                    if angle_luodian<=0.7:
                        cv2.circle(mask, (int(a),int(b)), 2,1,-1)

                        if ifdd:
                            x_dd, y_dd = a, b
                            ifdd=False

                        track_index.append(count)
                        track_location.append([int(a),int(b)])

            videoWriter.write(frame)

            # 迭代更新光流点信息
            diff = frame_diff(old_gray, frame_gray)
            old_gray = frame_gray.copy()
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=diff, **feature_params)
            print('tracking object-----> fire_index_start:',fire_index, 'fire_index_end:',fire_index_max, 'cur_index:',count)
            count += 1



        # &&&&&&&&&&  第三阶段：火焰出现当前时刻，通过之前所追踪的轨迹信息结合火焰信息得到完整的轨迹信息以及落点  &&&&&&&&&&
        elif count == fire_index-1:
            track_index.append(count)
            track_location.append([int(x_fire),int(y_fire)])
            cv2.circle(mask, (int(x_fire),int(y_fire)), 2,1,-1)
            cv2.imwrite(opticalpoints_filename, mask*255)

            mask_track = np.zeros_like(old_frame)
            # 将光流追踪点最上方的点和火焰边框中心点连线作为XX运行轨迹
            mask_track = cv2.line(mask_track, (int(x_dd),int(y_dd)), (int(x_fire), int(y_fire)), (0,255,0), 4)
            mask_track = cv2.circle(mask_track, (int(x_fire), int(y_fire)), 70, (0,0,255), -1)

            result_img = frame
            result_img = cv2.add(result_img, mask_track)
            videoWriter.write(result_img)

            print('finding location-----> fire_index_start:',fire_index, 'fire_index_end:',fire_index_max, 'cur_index:',count)
            count += 1
        


        # &&&&&&&&&&  第四阶段：火焰出现之后的所有帧  &&&&&&&&&&
        # 火焰开始出现，对火焰进行可视化
        elif count > fire_index-1:
            mask_fire = np.zeros_like(old_frame)

            # fire_local : [(tlx,tly),(brx,bry),w,h]
            if count in fire_local.keys():
                try:

                    tlx, tly, brx, bry, w, h = fire_local.get(count)[0][0], fire_local.get(count)[0][1], fire_local.get(count)[1][0], fire_local.get(count)[1][1],\
                                                fire_local.get(count)[2], fire_local.get(count)[3]

                    fire_range_1 = oneeye(tlx, tly) # BUG
                    fire_range_2 = oneeye(brx, bry)

                    fire_range_temp = []
                    fire_range_temp.append(fire_range_1[0])
                    fire_range_temp.append(fire_range_1[1])
                    fire_range_temp.append(fire_range_2[0])
                    fire_range_temp.append(fire_range_2[1])
                    fire_range_temp = np.array(fire_range_temp)
                    fire_range.update({count : [[tlx, tly, brx, bry],fire_range_temp, w, h]})
                    mask_fire = cv2.rectangle(mask_fire, fire_local.get(count)[0], fire_local.get(count)[1], (0,255,0), 5)
                except:
                    mask_fire = np.zeros_like(old_frame)

            result_img = frame
            result_img = cv2.add(result_img, mask_track)
            result_img = cv2.add(result_img, mask_fire)
            videoWriter.write(result_img)
            print('visualizing fire-----> fire_index_start:',fire_index, 'fire_index_end:',fire_index_max, 'cur_index:',count)
            count += 1



        # &&&&&&&&&&  第五阶段：火焰出现之后的所有帧  &&&&&&&&&&
        elif count>fire_index_max:
            print('cur_index>fire_index_end, ending-----> fire_index_start:',fire_index, 'fire_index_end:',fire_index_max, 'cur_index:',count)
            cap.release()
            videoWriter.release()
            return np.array(track_index), np.array(track_location), save_filename, fire_range, x_fire, y_fire

    print('ending-----> fire_index_start:',fire_index, 'fire_index_end:',fire_index_max, 'cur_index:',count) 
    cap.release()
    videoWriter.release()
    return np.array(track_index), np.array(track_location), save_filename, fire_range, x_fire, y_fire
