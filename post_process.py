
import re
import os
import datetime
import glob
from detect_api2 import detect_multi_videos
def check_string(re_exp, str):
    res = re.search(re_exp, str)
    if res:
        return True
    else:
        return False
class V:
    def __init__(self, video_dir) -> None:
        self.video_dir = video_dir
        # self.video_info # 列表，[{'name':0721, "one":{"path": d:/, "info": { "fire_info"}}; "two":{"path": d:/, "info": { "fire_info"}}；}]
        self.search_video_info()
        pass

    def post_process(self):
        # 1 多视频火焰检测
        print("1 启动yolo后处理")
        self.video_info = detect_multi_videos(self.video_info)

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
            # if  check_string('^[0-9]{8}_[0-9]{6}$',time_)  and datetime.datetime.strptime(time_,'%Y%m%d_%H%M%S').date() == _today.date():
            if  check_string('^[0-9]{8}_[0-9]{6}$',time_):

                videos_info.append({'time':str(time_)})
                pass
        
        # 2 获取每个时间段的视频
        for time_ in videos_info:
            time_dir = os.path.join(videos_path , time_['time']) 
            fileExtensions = [ "avi","mp4" ]
            videos_list = []
            for extension in fileExtensions: # 获取所有视频的相对路径
                videos_list.extend( glob.glob( time_dir + '\\*.' + extension  ))

            for v in videos_list: # 查看是否所有视频的命名都符合规则: 必须存在 ip
                if len(re.findall( r'[0-9]+(?:\.[0-9]+){3}',v)) > 0:
                    time_[re.findall( r'[0-9]+(?:\.[0-9]+){3}',v)[0]] = {}  # ip 作为键 
                    time_[re.findall( r'[0-9]+(?:\.[0-9]+){3}',v)[0]]['path'] = v
                    time_[re.findall( r'[0-9]+(?:\.[0-9]+){3}',v)[0]]['info'] = {}
        self.video_info = videos_info

video_dir = "video_new"
v = V(video_dir)
v.post_process()
a = 1
