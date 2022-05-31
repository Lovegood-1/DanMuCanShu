from cv2 import CALIB_ZERO_DISPARITY
import numpy as np
import torch
from models.reconstruction import RTStereoNet
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import argparse
from utils.torch_utils import cal_distance, cal_xyz_from_uv

class Estimate_disparty:
    def __init__(self) -> None:
        
        parser = argparse.ArgumentParser(description='PSMNet')
        parser.add_argument('--KITTI', default='2015',
                            help='KITTI version')
        parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                            help='select model')
        parser.add_argument('--loadmodel', default=r'pretrain\pretrained_Kitti2015_realtime.tar',
                            help='loading model')
        parser.add_argument('--leftimg', default= './VO04_L.png',
                            help='load model')
        parser.add_argument('--rightimg', default= './VO04_R.png',
                            help='load model')                                      
        parser.add_argument('--model', default='RTStereoNet',
                            help='select model')
        parser.add_argument('--maxdisp', type=int, default=192,
                            help='maxium disparity')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='enables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--source', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--weights',   default='best.pt', metavar='S',
                            help='random seed (default: 1)')
        self.args = parser.parse_args()
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)


        self.model = RTStereoNet(self.args.maxdisp)
 

        if self.args.cuda:
            self.model.cuda()

        if self.args.loadmodel is not None:
            print('load model')
            state_dict = torch.load(self.args.loadmodel)
            self.model.load_state_dict(state_dict['state_dict'])

        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in self.model.parameters()])))

    def calculate_disparty(self, imgr, imgl):
        disp = self.main(imgr, imgl)
        return disp

    def main(self, img_l, img_r):
        normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}
        infer_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(**normal_mean_var)])    

        # imgL_o = Image.open(path_l).convert('RGB')
        imgL_o = Image.fromarray(img_l).convert('RGB')
        # imgR_o = Image.open(path_r).convert('RGB')
        imgR_o = Image.fromarray(img_r).convert('RGB')

        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o) 
        

        # pad to width and hight to 16 times
        if imgL.shape[1] % 16 != 0:
            times = imgL.shape[1]//16       
            top_pad = (times+1)*16 -imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16                       
            right_pad = (times+1)*16-imgL.shape[2]
        else:
            right_pad = 0    

        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

        pred_disp = self.test(imgL,imgR)

        
        if top_pad !=0 and right_pad != 0:
            img = pred_disp[top_pad:,:-right_pad]
        elif top_pad ==0 and right_pad != 0:
            img = pred_disp[:,:-right_pad]
        elif top_pad !=0 and right_pad == 0:
            img = pred_disp[top_pad:,:]
        else:
            img = pred_disp
        return img

    def test(self, imgL,imgR):
    
        self.model.eval()

        if self.args.cuda:
           imgL = imgL.cuda()
           imgR = imgR.cuda()     

        with torch.no_grad():
            disp = self.model(imgL,imgR)

        disp = torch.squeeze(disp)
        pred_disp = disp.data.cpu().numpy()

        return pred_disp

    def get_object_xy(self, bbox  ,  disp, Q):
        """
        返回四个点的像素坐标，H，W
        """
        bbox = self.shrink_bbox_my(bbox).astype(np.int64)
        d3_dict = {}
        d3_list = []
        for i in list(range(bbox[0,0],bbox[1,0])):
            for j in  list(range(bbox[0,1],bbox[1,1])):
                d3_list.append([disp[j, i], i, j])
                d3_dict[(i,j)] = [disp[j, i], i, j]
        # 依据三维坐标，获取属于前景的像素[x,y]
            # 依据z，获得 80% 的分位数
        d3_list = np.array(d3_list)
        thre1, thre2 =  np.percentile(d3_list[:,0],10) ,  np.percentile(d3_list[:,0],90)# 取前10 的值作为第一个阈值，后10% 的值作为第二个阈值，只用这两个之间的
        valid_index = (d3_list[:,0] > thre1) * (d3_list[:,0] < thre2) + 0.0
        delte_index = [i  for i, v in   enumerate(list(range(d3_list.shape[0]))) if valid_index[i] < 0.1]
        print(d3_list.shape)
        d3_list = np.delete(d3_list,delte_index,axis=0)
        print(d3_list.shape)
        # 获取最左边或者最右边，最上面，最下面的点
        top_uv = d3_list[np.argmin(d3_list[:,-1])]
        bot_uv = d3_list[np.argmax(d3_list[:,-1])]
        right_uv = d3_list[np.argmax(d3_list[:,-2])]
        left_uv = d3_list[np.argmin(d3_list[:,-2])]
        top = cal_xyz_from_uv(top_uv[-2], top_uv[-1],disp, Q)
        # bot = cal_xyz_from_uv(bot_uv[-2], bot_uv[-1], disp, Q) # 一种计算方法
        bot = cal_xyz_from_uv(top_uv[-2], bot_uv[-1], disp, Q) # 第二种计算方法
        bot_uv[-2] = top_uv[-2]
        right = cal_xyz_from_uv(right_uv[-2],right_uv[-1], disp, Q)
        left = cal_xyz_from_uv(left_uv[-2], left_uv[-1], disp, Q)
        H = cal_distance(top, bot)

        W = cal_distance(right, left)
        print(H, W)
        return [top_uv, bot_uv, right_uv, left_uv], [H, W]

    def shrink_bbox_my(self, bbox, R = 0.8):
        """
        自己实现的用于缩放 bbox。步骤
            1. 计算长度
            2. 计算缩放长度
            3. (长度 - 缩放长度) /2
        """
        # 对于 x
        bbox = bbox.copy()
        x0, x1 = bbox[0,0], bbox[1,0]
        x0 , x1 = min(x0,x1), max(x0,x1)
        length = np.abs(x0 - x1)
        length_shrink = length * R
        x0 +=  (length - length_shrink) / 2
        x1 -=  (length - length_shrink) / 2
        bbox[0,0], bbox[1,0] = x0, x1
        # 对于 y
        x0, x1 = bbox[0,1], bbox[1,1]
        x0 , x1 = min(x0,x1), max(x0,x1)
        length = np.abs(x0 - x1)
        length_shrink = length * R
        x0 +=  (length - length_shrink) / 2
        x1 -=  (length - length_shrink) / 2
        bbox[0,1], bbox[1,1] = x0, x1    
        return bbox