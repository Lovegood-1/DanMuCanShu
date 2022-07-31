
import re
import os
import datetime
import glob
from detect_api2 import detect_multi_videos
from optflow_diff_api import video_process_multiple
from calibration.calibration_utils import PNPSolver, GetDistanceOf2linesIn3D
from utils.torch_utils import get_file_list
def check_string(re_exp, str):
    res = re.search(re_exp, str)
    if res:
        return True
    else:
        return False
class V:
    def __init__(self, video_dir, calibration_dir) -> None:
        self.video_dir = video_dir
        self.calibration_dir = calibration_dir
        # self.video_info # 列表，[{'name':0721, "one":{"path": d:/, "info": { "fire_info"}}; "two":{"path": d:/, "info": { "fire_info"}}；}]
        self.search_video_info()
        self.search_camera_info()
        
        pass

    def post_process(self):
        # 1 多视频火焰检测
        print("1 启动yolo后处理")
        self.video_info = detect_multi_videos(self.video_info) # 火焰检测
        self.video_info = video_process_multiple(self.video_info, self.camera_info) # 导弹轨迹
        pass

    def search_video_info(self):
        """
        初始化视频列表
            [{'time':'20220727_111040', 
              '192.168.1.1':
                    {"path": d:/, 
                    "info": { "dict_all": dict;  # 这个是火焰检测负责计算
                              "dict1_first_fire": 
                              "missle_info":}     # 这个导弹检测计算
              '192.168.1.2':
                    {"path": d:/, 
                    "info": { "dict_all": dict;  # 这个是火焰检测负责计算
                              "dict1_first_fire":
                              "missle_info":}     # 这个导弹检测计算
                              }]  
        """
        videos_info = []
        videos_path = self.video_dir
        # 1 初始化时间并加入
        _today = datetime.datetime.now()
        time_list = os.listdir(videos_path) 
        for time_ in time_list: # 如果符合命名规则而且是今天则加入
            # if  check_string('^[0-9]{8}_[0-9]{6}$',time_)  and datetime.datetime.strptime(time_,'%Y%m%d_%H%M%S').date() == _today.date(): TODO: 这里运行的时候一定要加
            if  check_string('^[0-9]{8}_[0-9]{6}$',time_):

                videos_info.append({'time':str(time_)})
                videos_info.append({'result':{}})
                pass
        
        # 2 获取每个时间段的视频
        for time_ in videos_info:
            time_dir = os.path.join(videos_path , time_['time']) 
            fileExtensions = [ "avi","mp4" ]
            videos_list = []
            for extension in fileExtensions: # 获取所有视频的相对路径
                videos_list.extend( glob.glob( time_dir + '\\*.' + extension  ))

            for v in videos_list: # 查看是否所有视频的命名都符合规则: 必须存在 ip
                ip_ = re.findall( r'[0-9]+(?:\.[0-9]+){3}',v)[0]
                if len(ip_) > 0:
                    time_[ip_] = {}  # ip 作为键 
                    time_[ip_]['path'] = v
                    time_[ip_]['info'] = {}
        self.video_info = videos_info


    def search_camera_info(self):
        calibration_dir = self.calibration_dir
        self.camera_info = {} # {'192.168.1.1':PNPSolver, '192.168.1.2':PNPSolver,}
        camera_list = get_file_list(calibration_dir, ['mat']) # 获取所有 mat 文件
        for camera in camera_list:
            ip_ = re.findall( r'[0-9]+(?:\.[0-9]+){3}',camera)[0]
            if len(ip_) > 0:
                assert os.path.isfile(camera)
                assert  os.path.isfile(os.path.join(calibration_dir,  str(ip_) + '.csv'))
                self.camera_info[ip_] = PNPSolver(camera, os.path.join(calibration_dir,  str(ip_) + '.csv'))# TODO: 这里有两个文件，以后最好是一个


    def write_reuslt_to_txt(self):
        self.video_info = {}
        pass



video_dir = "video_new"
v = V(video_dir, r'calibration\calibration_mat')
v.post_process()
v.write_reuslt_to_txt()
a = 1