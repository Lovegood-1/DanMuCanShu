import cv2
import numpy as np
import scipy
import scipy.linalg
import pandas as pd
import os



path1='neican.csv'
path2='waican.csv'

data1=pd.read_csv(path1)
data = pd.read_csv(path2)
pixel_cor1 = data[['u1','v1']]
pixel_cor1 = np.array(pixel_cor1).reshape(-1,1,2)
pixel_cor1 = pixel_cor1.astype(np.float32)

pixel_cor2 = data[['u2','v2']]
pixel_cor2 = np.array(pixel_cor2).reshape(-1,1,2)
pixel_cor2 = pixel_cor2.astype(np.float32)

worldpoint1 = data[['x1','y1','z1']]
worldpoint1 = np.array(worldpoint1,dtype=np.float32)

worldpoint2 = data[['x2','y2','z2']]
worldpoint2 = np.array(worldpoint2,dtype=np.float32)

Intrin_M1=data1[['m1','m2','m3']]
Intrin_M1 = np.array(Intrin_M1)
Intrin_M2=data1[['m4','m5','m6']]
Intrin_M2 = np.array(Intrin_M2)
dist1=data1[['r1']]
dist1 = np.array(dist1)
dist1 = np.row_stack((dist1,[0]))
dist2=data1[['r2']]
dist2 = np.array(dist2)
dist2 = np.row_stack((dist2,[0]))


zer0=np.array([[0],[0], [0]],dtype=np.float64)#0列向量
zer1=np.array([0,0,0],dtype=np.float64)#0行向量
pingyi=np.array([1])
K1 = Intrin_M1
K2=Intrin_M2
objPoints1 = worldpoint1
objPoints2 = worldpoint2
imgPoints1 = pixel_cor1
imgPoints2 = pixel_cor2

_, rvec1, tvec1, inliers  = cv2.solvePnPRansac(objPoints1, imgPoints1, K1, dist1)
_, rvec2, tvec2, inliers  = cv2.solvePnPRansac(objPoints2, imgPoints2, K2, dist2)
Rvce1,_=cv2.Rodrigues(rvec1)
Rvce2,_=cv2.Rodrigues(rvec2)
#15号摄像头
K1=np.concatenate((K1,zer0),axis=1)#内参矩阵3*4
R_Rvce1=np.row_stack((Rvce1,zer1))#旋转矩阵4*3
R_tvec1=np.row_stack((tvec1,pingyi))#平移向量4*1
R_M=np.concatenate((R_Rvce1,R_tvec1),axis=1)#外参4*4
final_Matrix15=np.dot(K1,R_M)#系数矩阵
#27号摄像头
K2=np.concatenate((K2,zer0),axis=1)#内参矩阵3*4
R_Rvce2=np.row_stack((Rvce2,zer1))#旋转矩阵4*3
R_tvec2=np.row_stack((tvec2,pingyi))#平移向量4*1
R_M=np.concatenate((R_Rvce2,R_tvec2),axis=1)#外参4*4
final_Matrix27=np.dot(K2,R_M)#系数矩阵

def rebuild(u1,v1,u2,v2):
      #图1
     left_=([final_Matrix15[0,3]-u1*final_Matrix15[2,3]],
               [final_Matrix15[1,3]-v1*final_Matrix15[2,3]])
     right_=(u1*final_Matrix15[2,0:3]-final_Matrix15[0,0:3],
                v1*final_Matrix15[2,0:3]-final_Matrix15[1,0:3])
#图2
     left_2=([final_Matrix27[0,3]-u2*final_Matrix27[2,3]],
               [final_Matrix27[1,3]-v2*final_Matrix27[2,3]])
     right_2=(u2*final_Matrix27[2,0:3]-final_Matrix27[0,0:3],
                v2*final_Matrix27[2,0:3]-final_Matrix27[1,0:3])
#求解方程
     Left=np.row_stack((left_,left_2))
     Right=np.row_stack((right_,right_2))
     ATA=np.dot(Right.T,Right)
     ATb=np.dot(Right.T,Left)
     point=np.linalg.solve(ATA,ATb)
     return point




def null(A,eps = 1e-15):
    u, s, vh = scipy.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)

    return scipy.transpose(null_space)


def oneeye(u,v):
#使用一个摄像头时需要注释掉零一个
     final_M = final_Matrix15#使用15号摄像头单目
     final_M = final_Matrix27#使用27号摄像头单目

# 1 构建左矩阵，2 * 1
     left = np.array([[final_M[0,3] - u * final_M[2,3]],
     [final_M[1,3] - v * final_M[2,3]]])
     right = np.array([u * final_M[2,0:3] - final_M[0,0:3],
     v * final_M[2,0:3] - final_M[1,0:3]])
# 2 最后坐标 cor = x,y, cor 对应于 matlab 中的 final_word_cor
     point= np.linalg.solve(right[:,:2], left)
     return point



if __name__ == '__main__':
 #finalpoint=rebuild(u1=436,v1=445,u2=319,v2=180)/1000
 finalpoint1 = oneeye(2363,1409) / 1000
 finalpoint = rebuild(2363, 1409,200,500) / 1000
 print(finalpoint1)
 print(finalpoint)



