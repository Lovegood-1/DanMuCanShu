"""
自定义缓存保存 RTSP 视频
"""
import os
import cv2
import gc
import time
from multiprocessing import Process, Manager
 
 
# 向共享缓冲栈中写入数据:
def write(stack, cam, top: int) -> None:
    print('Process to write: %s' % os.getpid())
    cap = cv2.VideoCapture(cam)
    while True:
        _, img = cap.read()
        if _:
            stack.append(img)
            # 每到一定容量清空一次缓冲栈
            # 利用gc库，手动清理内存垃圾，防止内存溢出
            if len(stack) >= top:
                del stack[::2]
                gc.collect()
 
 
# 在缓冲栈中读取数据:
def read(stack) -> None: #提醒返回值是一个None
    print('Process to read: %s' % os.getpid())
    index = 0
    fourcc = cv2.VideoWriter_fourcc('P','I','M','2')   #MPEG-4.2
    #下面的格式根据不同情况选择
    # fourcc = cv2.VideoWriter_fourcc('P','I','M','1')  # MPEG-1
    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G') # motion-jpeg codec
    # fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', '3') # MPEG-4.3 codec
    # fourcc = cv2.VideoWriter_fourcc('P', 'I', 'M', '2')  # MPEG-4.2
    # fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X') # MPEG - 4    codec
    # fourcc = cv2.VideoWriter_fourcc('U', '2', '6', '3') # H263    codec
    # fourcc = cv2.VideoWriter_fourcc('I', '2', '6', '3') # H263I   codec
    # fourcc = cv2.VideoWriter_fourcc('F', 'L', 'V', '1') # FLV1 codec
 
 
    out = cv2.VideoWriter('video_out.mpg',fourcc , 6.3, (1280, 720))
 
    start_time = time.time()
    x = 1  # displays the frame rate every 1 second
    counter = 0
 
    print("开始逐帧读取")
    while True:
        # print("正在读取第%d帧：" %index)
        if len(stack) >= 5:
            frame = stack.pop(0)
 
            index = index+1 
 
            #直接保存视频
            out.write(frame)
            cv2.imshow("img", frame)
 
            #计算fps
            counter += 1
            if (time.time() - start_time) > x:
 
                print("FPS: ", counter / (time.time() - start_time))
                counter = 0
                start_time = time.time()
 
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        else:
            continue
    out.release()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    # 父进程创建缓冲栈，并传给各个子进程：
    q = Manager().list()
    pw = Process(target=write, args=(q, "rtsp://192.168.4.101/video1", 100)) 
    pr = Process(target=read, args=(q,))
    pw.start()
    pr.start()
 
    pr.join()
 
    pw.terminate()