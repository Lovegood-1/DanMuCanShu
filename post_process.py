class V:
    def __init__(self, video_dir) -> None:
        self.video_dir = video_dir
        self.video_info # 列表，[{'name':0721, "one":{"path": d:/, "info": { "fire_info"}}; "two":{"path": d:/, "info": { "fire_info"}}；}]
        self.search_video_info()
        pass

    def post_process(self):
        # 1 多视频火焰检测


        pass

    def search_video_info(self):
        """初始化视频列表
            [{'time':20220727_111040, 
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
        pass

    
video_dir = r''
v = V(video_dir)
