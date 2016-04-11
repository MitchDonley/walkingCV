__author__ = 'brycerich'
import cv2
import numpy as np
img = cv2.imread('jurassic_world.jpg')
from time import sleep

height, width, depth = img.shape
go = True
print 'go is true'
cv2.imshow("image", img)
while go:
    print "in while"
    k = cv2.waitKey(0)& 0xFF
    if k == 27:         # wait for ESC key to exit
        go = False
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        while go:
            for i in range (0, height):
                for j in range (0, width):
                    # img.itemset((i,j,2), (img.item(i,j,2) + 127)% 255)
                    # img.itemset((i,j,1), (img.item(i,j,1) + 127)% 255)
                    # img.itemset((i,j,0), (img.item(i,j,0) + 127)% 255)

                    # img.itemset((i,j,2), (255 - img.item(i,j,2)) % 255)
                    # img.itemset((i,j,1), (255 - img.item(i,j,1)) % 255)
                    # img.itemset((i,j,0), (255 - img.item(i,j,0)) % 255)
                    r = img.item(i,j,2)
                    g = img.item(i,j,1)
                    b = img.item(i,j,0)

                    img.itemset((i,j,2), (b*2) % 255)
                    img.itemset((i,j,1), (r*2) % 255)
                    img.itemset((i,j,0), (g*2) % 255)

                    # img.itemset((i,j,2), (r + 10) % 255 )
                    # img.itemset((i,j,1), (g + 10) % 255 )
                    # img.itemset((i,j,0), (b + 10) % 255 )

            cv2.imshow('image',img)
            sleep(1)
            k = cv2.waitKey(1)& 0xFF
            if k == 27:         # wait for ESC key to exit
                go = False
                cv2.destroyAllWindows()
            else: #k == ord('s'): # wait for 's' key to save and exit
                go = True
        go = True
    elif k == ord('f'):
        face1 = img[:]
        # face2 = []
        # face1[:] = img[100:150,100:150]
        # face2[:] = img[200:250,100:150]
        img[60:95,224:250] = face1[75:110,120:146]
        # img[100:150,100:150] = face1[200:250,100:150]
        cv2.imshow("image", img)
