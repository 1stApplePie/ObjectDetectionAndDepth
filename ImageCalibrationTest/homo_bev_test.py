import cv2
import numpy as np

image_width = 640
image_height = 480
capture = cv2.VideoCapture(3)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
calibrate_mtx = np.array([
    [374.839208, 0.0, 329.917309],
    [0.0, 371.474881, 244.814906],
    [0.0, 0.0, 1.0]
    ])

dist = np.array([-0.299279, 0.067634, -0.001693, -0.002189, 0.0])

cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(calibrate_mtx, dist,
                            (image_width, image_height), 1, (image_width, image_height))
x, y, w, h = cal_roi
map1, map2 = cv2.initUndistortRectifyMap(calibrate_mtx, dist, None, cal_mtx, (image_width, image_height), cv2.CV_32FC1)

point_src = np.array([
    214, 253,   # 0, 0
    251, 253,   # 0, 1
    288, 254,   # 0, 2
    328, 254,   # 0, 3
    368, 255,   # 0, 4
    408, 254,   # 0, 5
    237, 256,   # 1, 1
    281, 257,   # 1, 2
    327, 259,   # 1, 3
    377, 258,   # 1, 4
    423, 258,   # 1, 5
    216, 260,   # 2, 1
    272, 262,   # 2, 2
    328, 262,   # 2, 3
    389, 263,   # 2, 4
    445, 263,   # 2, 5
    256, 269,   # 3, 2
    328, 270,   # 3, 3
    408, 270,   # 3, 4
    225, 281,   # 4, 2
    329, 286,   # 4, 3
    441, 284,   # 4, 4
    148, 315,   # 5, 2
    329, 320,   # 5, 3
    526, 318,   # 5, 4
    239, 464    # LiDAR
], dtype=float)
point_src = point_src.reshape(26, 2)

point_dst = np.array([
    0, 0,
    90, 0,
    180, 0,
    270, 0,
    360, 0,
    450, 0,
    90, 90,
    180, 90,
    270, 90,
    360, 90,
    450, 90,
    90, 180,
    180, 180,
    270, 180,
    360, 180,
    450, 180,
    180, 270,
    270, 270,
    360, 270,
    180, 360,
    270, 360,
    360, 360,
    180, 450,
    270, 450,
    360, 450,
    270, 540
], dtype=float)
point_dst = point_dst.reshape(26, 2)

homography_matrix, status = cv2.findHomography(point_src, point_dst, cv2.LMEDS)

def calibrate_image(src):
    global image_width, image_height
    dst = cv2.remap(src, map1, map2, cv2.INTER_LINEAR)
    dst = dst[y : y + h, x : x + w]
    return cv2.resize(dst, (image_width, image_height))

while capture.isOpened():
    _, src = capture.read()
    cal = calibrate_image(src)

    # cal = cv2.imread("C:/Users/HP/Desktop/ws/python/bev-test/nonpointed_domino_image.png")
    
    bev_image = cv2.warpPerspective(cal, homography_matrix, (540, 540), flags = cv2.INTER_CUBIC)

    for i in range(7):
        for j in range(7):
            cv2.circle(bev_image, (i * 90, j * 90), 2 , (255, 0, 0), -1)
        
    cv2.imshow("cal", cal)
    cv2.imshow("bev", bev_image)
        # cv2.imwrite("bev_image-LMEDS-CUBIC.png", bev_image)

    if cv2.waitKey(10) == 27:
        capture.release()
        break