import numpy as np
import cv2
import multiprocessing as mp
import Queue

def draw_arrow(image, p, q, color, arrow_magnitude=9, thickness=3, line_type=8, shift=0):
    # adapted from http://mlikihazar.blogspot.com.au/2013/02/draw-arrow-opencv.html

    # draw arrow tail
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # calc angle of the arrow
    angle = np.arctan2(p[1]-q[1], p[0]-q[0])
    # starting point of first line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
    # draw first half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # starting point of second line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
    # draw second half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)

if __name__ == '__main__':

    print cv2.__version__

    #cap = cv2.VideoCapture('walking_techwalkway.MOV')
    #cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('Gondola.mp4')
    cap = cv2.VideoCapture('GOPR0133_480.mov')

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
    dimensions = old_frame.shape
    frame_height = dimensions[0]
    frame_width = dimensions[1]

    section_counts = [0]*9
    cutdown = [False]*9
    isEven = [False]*9
    print section_counts


    # Create a mask image for drawing purposes
    # mask = np.zeros_like(old_frame)
    maskq = mp.Queue()
    past10 = []
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
        print p1
        print p1.shape
        num = p1.shape[0]
        for i in range (0,num):
            if p1[i][0][0] < frame_width/3:
                column = 0;
            elif p1[i][0][0] < (frame_width * 2)/3:
                column = 1;
            else:
                column = 2;

            if p1[i][0][1] < frame_height/3:
                row = 0;
            elif p1[i][0][1] < (frame_height * 2)/3:
                row = 1;
            else:
                row = 2;
            if cutdown[column*3 + row] is True and i % 2 == 1:
                st[i] = 0

        good_new = p1[st == 1]
        print "printing st"
        print st
        print "printing p1"
        print p1
        print "printing good new"
        print good_new
        section_counts = [0]*9
        for i in good_new:
            if i[0] < frame_width/3:
                column = 0;
            elif i[0] < (frame_width * 2)/3:
                column = 1;
            else:
                column = 2;

            if i[0] < frame_height/3:
                row = 0;
            elif i[0] < (frame_height * 2)/3:
                row = 1;
            else:
                row = 2;
            section_counts[column*3 + row] +=1
        print "printing section counts"
        print section_counts
        print good_new.shape
        for i in range(0,9):
            if section_counts[i] > 15:
                cutdown[i] = True
            else:
                cutdown[i] = False
        print "printing cutdown"
        print cutdown
        print "printing is even"
        print isEven


        #print(p0)
        #print(p1)
        #print(st)
        good_old = p0[st == 1]
        # for i in st:
            # if i == 0:
                # print("done")

        # draw the tricks
        diff = []
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            for j in range(0,10):
                mask = maskq.get()
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                maskq.put(mask)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
            diff.append([a-c,b-d])
        mask = maskq.get()
        img = cv2.add(frame,mask)
        mask = np.zeros_like(old_frame)
        maskq.put(mask)
        #print(diff)
        xdiff = 0
        ydiff = 0
        for i in diff:
            xdiff += i[0]
            ydiff += i[1]
        xdiff = xdiff/diff.__len__()
        ydiff = ydiff/diff.__len__()
        if xdiff > 0:
            phrase = "right"
        else:
            phrase = "left"
        if ydiff > 0:
            phrase += "-up"
        else:
            phrase += "-down"
        print xdiff,ydiff,phrase
        if(past10.__len__() < 10):
            past10.append([xdiff,ydiff])
        else:
            past10[timer%10] = [xdiff,ydiff]
        xavg = 0
        yavg = 0
        for i in past10:
            xavg += i[0]
            yavg += i[1]
        xavg /= past10.__len__()
        yavg /= past10.__len__()

        draw_arrow(img,(150,237),(150+int(xdiff * -10),237+int(ydiff * -10)),(255,0,0))
        draw_arrow(img,(150,237),(150+int(xavg * -10),237+int(yavg * -10)),(0,255,0))

        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        #print(p0.size)

        if timer >= 20:
            # while p1 is None:
            #     p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # while p0 is None:
            # print(p0)
            # print type(p0)
            p01 = cv2.goodFeaturesToTrack(old_gray, mask = None,useHarrisDetector= True, **feature_params)
            # print(p01)
            # print type(p01)
            p02 = p0.ravel().reshape(-1,1,2)
            # print p02
            for i in p01:
                found = False
                for j in p02:
                    j2 = np.around(j)
                    # print("values")
                    # print i
                    # print j2
                    # print"end values"
                    # print i[0][0] + 10
                    # print j2[0][0]
                    # print i[0][0] - 10
                    # print i[0][1] + 10
                    # print j2[0][1]
                    # print i[0][1] - 10
                    if((i[0][0] > j2[0][0] - 5 and i[0][0] < j2[0][0] + 5) and (i[0][1] > j2[0][1] - 5 and i[0][1] < j2[0][1] + 5)):
                        found = True
                        # print"true"
                        break
                    # print "false"
                if found == False:
                    # print p02
                    p02 = np.append(p02,i)
                    # print p02
                    p02 = p02.ravel().reshape(-1,1,2)
            #p02 = np.append(p0.ravel(), p01.ravel())
            # print p02
            p02 = p02.reshape(-1,1,2)
            p0 = p02
            # print(p0.size)
            mask = np.zeros_like(old_frame)
            timer = 0


        timer += 1
        #print(timer)

    cv2.destroyAllWindows()
    cap.release()