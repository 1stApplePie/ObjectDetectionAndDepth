import cv2
import numpy as np

def callback(x):
    pass

cap = cv2.VideoCapture(3)
cv2.namedWindow('image')

ilowH = 0
ihighH = 255

ilowL = 0
ihighL = 255
ilowS = 0
ihighS = 255

# create trackbars for color change
cv2.createTrackbar('lowH','image',ilowH,255,callback)
cv2.createTrackbar('highH','image',ihighH,255,callback)

cv2.createTrackbar('lowL','image',ilowL,255,callback)
cv2.createTrackbar('highL','image',ihighL, 255,callback)

cv2.createTrackbar('lowS','image',ilowS,255,callback)
cv2.createTrackbar('highS','image',ihighS, 255,callback)

while True:
    # grab the frame
    ret, frame = cap.read()

    # get trackbar positions
    ilowH = cv2.getTrackbarPos('lowH', 'image')
    ihighH = cv2.getTrackbarPos('highH', 'image')
    ilowL = cv2.getTrackbarPos('lowL', 'image')
    ihighL = cv2.getTrackbarPos('highL', 'image')
    ilowS = cv2.getTrackbarPos('lowS', 'image')
    ihighS = cv2.getTrackbarPos('highS', 'image')

    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    # h, l, s = cv2.split(hls)

    # _, t_h = cv2.inRange(h, ilowH, ihighH, cv2.THRESH_BINARY)
    # _, t_l = cv2.threshold(l, ilowL, ihighL, cv2.THRESH_BINARY)
    # _, t_s = cv2.threshold(s, ilowS, ihighS, cv2.THRESH_BINARY)


    # temp = cv2.bitwise_and(t_h, t_l)
    # frame = cv2.bitwise_and(temp, t_s)

    frame = cv2.inRange(hls, (ilowH, ilowS, ilowL), (ihighH, ihighS, ihighL))

    try:
        count = cv2.countNonZero(frame)

        print("Non Zero: ", count)
    except:
        print("pass")

    # show thresholded image
    cv2.imshow('image', frame)
    k = cv2.waitKey(10) & 0xFF # large wait time to remove freezing
    if k == 113 or k == 27:
        break