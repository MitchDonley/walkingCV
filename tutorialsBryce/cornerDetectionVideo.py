__author__ = 'brycerich'

import cv2
import numpy as np

# k = cv2.waitKey(10000) & 0xff
# if k == ord('w'):
# cap = cv2.VideoCapture('walking_techwalkway.MOV')
# if k == ord('g'):
cap = cv2.VideoCapture('Gondola.mp4')
#cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(1):
    ret, frame1 = cap.read()
    next = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next,None,0.5,3,15,3,5,1.2,0)
    next = np.float32(next)
    print(flow)
    dst = cv2.cornerHarris(next,2,3,0.02)
    dst = cv2.dilate(dst, None)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    frame1[dst>0.01*dst.max()] = [255,0,0]
    #print(dst)

    cv2.imshow('dst',frame1)
    cv2.imshow('opticalflow', rgb)


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('cornerDetectfb.png',frame1)
        cv2.imwrite('cornerdetectdst.png',dst)
    prvs = next
cap.release()
cv2.destroyAllWindows()