__author__ = 'brycerich'
import numpy as np
import cv2

cap = cv2.VideoCapture("Gondola.mp4")
mog2 = cv2.createBackgroundSubtractorMOG2()
while(1):
    ret, frame = cap.read()
    mog2mask = mog2.apply(frame)
    newFrame = cv2.bitwise_and(frame, frame, mask = mog2mask)

    cv2.imshow('frame1',newFrame)
    # cv2.imshow('frame',frame)
    # cv2.imshow('Mog',mog2mask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
if subtractionMethod == 1:
    cap = cv2.VideoCapture("Gondola.mp4")

    fgbg = cv2.createBackgroundSubtractorGMG()

    while(1):
        ret, frame = cap.read()

        fgmask = fgbg.apply(frame)

        cv2.imshow('frame1',fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
elif subtractionMethod == 2:
    cap = cv2.VideoCapture('Gondola.mp4')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.createBackgroundSubtractorGMG()

    while(1):
        ret, frame = cap.read()

        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        newFrame = cv2.bitwise_and(frame, frame, mask = fgmask)

        cv2.imshow('frame1',newFrame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()