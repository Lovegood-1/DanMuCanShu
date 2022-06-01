import  calibration.param_0531 as stereoconfig
from utils.video_rectify import video_rectify_double

left_video = r'video\camera_rgb_double.avi'
config = stereoconfig.stereoCamera()
Video = video_rectify_double(left_video,  config)
Video.rectify_video_double()
Video.triangulation()