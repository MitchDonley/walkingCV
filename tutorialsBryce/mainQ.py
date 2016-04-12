import numpy as np
import cv2
import multiprocessing as mp
import Queue

print cv2.__version__

#cap = cv2.VideoCapture('walking_techwalkway.MOV')
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Gondola.mp4')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2500, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, useHarrisDetector= True, **feature_params)

# Create a mask image for drawing purposes
# mask = np.zeros_like(old_frame)
maskq = mp.Queue()
for i in range(0,10):
    maskq.put(np.zeros_like(old_frame))
timer = 0

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    while p1 is None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    print(p0)
    #print(p1)
    print(st)
    good_old = p0[st == 1]

    # draw the tricks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        for j in range(0,10):
            mask = maskq.get()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            maskq.put(mask)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    mask = maskq.get()
    img = cv2.add(frame,mask)
    mask = np.zeros_like(old_frame)
    maskq.put(mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

    if timer >= 20:
        # while p1 is None:
        #     p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # while p0 is None:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None,useHarrisDetector= True, **feature_params)
        mask = np.zeros_like(old_frame)
        timer = 0


    timer += 1
    print(timer)

cv2.destroyAllWindows()
cap.release()