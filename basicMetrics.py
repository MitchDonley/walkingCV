__author__ = 'brycerich'
import numpy as np
import cv2

cap = cv2.VideoCapture('walking_techwalkway.MOV')
print "Current time: %f" % (cap.get(0))
print "Current frame: %f" % (cap.get(1))
print "Fraction of way through video: %f" % (cap.get(2))
print "Frame width: %f" % (cap.get(3))
print "Frame height: %f" % (cap.get(4))
print "Fps: %f" % (cap.get(5))
print "Fourcc: %s" % (cap.get(6))
print cap.get(6)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    # print "Current time: %f" % (cap.get(0))
    # print "Current frame: %f" % (cap.get(1))
    # print "Fraction of way through video: %f" % (cap.get(2))
    # print "Frame width: %f" % (cap.get(3))
    # print "Frame height: %f" % (cap.get(4))
    # print "Fps: %f" % (cap.get(5))
    # print "Fourcc: %f" % (cap.get(6))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()