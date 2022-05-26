import cv2

path_ = 'D:\\files\\xihe_data\\to_test\\157.mp4'
video = cv2.VideoCapture(path_)

# get info
fps = video.get(cv2.CAP_PROP_FPS)
size = (
    int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
fNUMS = video.get(cv2.CAP_PROP_FRAME_COUNT)

# read frame
success, frame = video.read()
while success:
    cv2.setWindowTitle(
        "test",
        "Mytest"
    )
    frame = cv2.resize(frame, (960,540))
    cv2.imshow('windows', frame)
    print(fps)
    cv2.waitKey(int(1000 / int(fps)))
    success, frame = video.read()
video.release()