import shutil
import cv2
import numpy as np
import os
import pandas as pd

def camera2word(realstart_x, realstart_y, realstart_z):
    root = 'video_template\\' 

    import  calibration.param_0622 as stereoconfig
    from utils.video_rectify import video_rectify_double
    left_video = os.path.join(root,'camera_rgb_double.avi')
    config = stereoconfig.stereoCamera()
    Video = video_rectify_double(left_video,  config)
    rebuild = Video.triangulation

    path2='waican.csv' 
    data = pd.read_csv(path2)
    pixel_cor1 = data[['u1','v1']]
    pixel_cor1 = np.array(pixel_cor1).reshape(-1,1,2)
    pixel_cor1 = pixel_cor1.astype(np.float32)

    worldpoint1 = data[['x1','y1','z1']]
    worldpoint1 = np.array(worldpoint1,dtype=np.float32)

    zer0=np.array([[0],[0], [0]],dtype=np.float64)#0列向量
    zer1=np.array([0,0,0],dtype=np.float64)#0行向量
    pingyi=np.array([1])
    objPoints1 = worldpoint1 
    imgPoints1 = pixel_cor1

    _, rvec1, tvec1, inliers  = cv2.solvePnPRansac(objPoints1, imgPoints1, config.cam_matrix_left, config.distortion_l)
    # realstart_x, realstart_y, realstart_z = rebuild(247,250,806-640,251)
    camera_cor = np.array([realstart_x.item(), realstart_y.item(), realstart_z.item(),1]).reshape(4,1)

    # TODO : 外参矫正图
    Rvce1,_=cv2.Rodrigues(rvec1)
    R_Rvce1=np.row_stack((Rvce1,zer1))#旋转矩阵4*3
    R_tvec1=np.row_stack((tvec1,pingyi))#平移向量4*1
    R_M=np.concatenate((R_Rvce1,R_tvec1),axis=1)#外参4*4
    if 1==2: 
        img = cv2.imread('rec.png')
        nose_end_point2D, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rvec1, tvec1,  config.cam_matrix_left, config.distortion_l)
        nose_end_point2D2, jacobian = cv2.projectPoints(np.array([(-0.13956, -0.6019, -0.23849)]), rvec1, tvec1,  config.cam_matrix_left, config.distortion_l)
        nose_end_point2D3, jacobian = cv2.projectPoints(np.array([(-1.0, 0.0, 0.0)]), rvec1, tvec1,  config.cam_matrix_left, config.distortion_l)
        cv2.circle(img, (int(nose_end_point2D[0,0,0]), int(nose_end_point2D[0,0,1])), 3, (0,0,255), -1)
        cv2.circle(img, (int(nose_end_point2D2[0,0,0]), int(nose_end_point2D2[0,0,1])), 3, (0,255,255), -1)
        cv2.circle(img, (int(nose_end_point2D3[0,0,0]), int(nose_end_point2D3[0,0,1])), 3, (0,0,255), -1)
    word_cord =  np.linalg.inv(R_M) @ camera_cor

    return np.array(word_cord[0,0]), np.array(word_cord[0,0]), np.array(word_cord[0,0])
if __name__=='__main__':


    root = 'video\\' 

    import  calibration.param_0622 as stereoconfig
    from utils.video_rectify import video_rectify_double
    left_video = os.path.join(root,'camera_rgb_double.avi')
    config = stereoconfig.stereoCamera()
    Video = video_rectify_double(left_video,  config)
    rebuild = Video.triangulation

    path2='waican.csv' 
    data = pd.read_csv(path2)
    pixel_cor1 = data[['u1','v1']]
    pixel_cor1 = np.array(pixel_cor1).reshape(-1,1,2)
    pixel_cor1 = pixel_cor1.astype(np.float32)

    worldpoint1 = data[['x1','y1','z1']]
    worldpoint1 = np.array(worldpoint1,dtype=np.float32)

    zer0=np.array([[0],[0], [0]],dtype=np.float64)#0列向量
    zer1=np.array([0,0,0],dtype=np.float64)#0行向量
    pingyi=np.array([1])
    objPoints1 = worldpoint1 * 0.5
    imgPoints1 = pixel_cor1

    _, rvec1, tvec1, inliers  = cv2.solvePnPRansac(objPoints1, imgPoints1, config.cam_matrix_left, config.distortion_l)
    realstart_x, realstart_y, realstart_z = rebuild(247,250,806-640,251)
    camera_cor = np.array([realstart_x.item(), realstart_y.item(), realstart_z.item(),1]).reshape(4,1)

    # TODO : 外参矫正图
    Rvce1,_=cv2.Rodrigues(rvec1)
    R_Rvce1=np.row_stack((Rvce1,zer1))#旋转矩阵4*3
    R_tvec1=np.row_stack((tvec1,pingyi))#平移向量4*1
    R_M=np.concatenate((R_Rvce1,R_tvec1),axis=1)#外参4*4
    if 1==2: 
        img = cv2.imread('rec.png')
        nose_end_point2D, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rvec1, tvec1,  config.cam_matrix_left, config.distortion_l)
        nose_end_point2D2, jacobian = cv2.projectPoints(np.array([(-0.13956, -0.6019, -0.23849)]), rvec1, tvec1,  config.cam_matrix_left, config.distortion_l)
        nose_end_point2D3, jacobian = cv2.projectPoints(np.array([(-1.0, 0.0, 0.0)]), rvec1, tvec1,  config.cam_matrix_left, config.distortion_l)
        cv2.circle(img, (int(nose_end_point2D[0,0,0]), int(nose_end_point2D[0,0,1])), 3, (0,0,255), -1)
        cv2.circle(img, (int(nose_end_point2D2[0,0,0]), int(nose_end_point2D2[0,0,1])), 3, (0,255,255), -1)
        cv2.circle(img, (int(nose_end_point2D3[0,0,0]), int(nose_end_point2D3[0,0,1])), 3, (0,0,255), -1)
    
    a = 1
    