import  calibration.param_0531 as stereoconfig
from utils.video_rectify import video_rectify_double
import math
def cal_distance(p1,p2):
    return math.sqrt(math.pow(p2[0] - p1[0],2) + math.pow(p2[1] - p1[1],2) + math.pow(p2[2] - p1[2],2))
left_video = r'video\camera_rgb_double.avi'
config = stereoconfig.stereoCamera()
Video = video_rectify_double(left_video,  config)
Video.rectify_video_double()

# Next: triangulation
# pixel1:
xl, yl  , xr, yr = 332,290,306,295
xyz = Video.triangulation(xl, yl  , xr, yr)
# pixel2:
xl, yl  , xr, yr = 396,304,373,307
xyz2 = Video.triangulation(xl, yl  , xr, yr)
d = cal_distance(xyz,xyz2)
a = 1

