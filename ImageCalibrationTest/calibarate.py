import cv2, time
import numpy as np

warp_image_width = 640
warp_image_height = 480

warp_x_margin = 0
warp_y_margin = 0

lane_bin_th = 145

pts1_x, pts1_y = 200 - warp_x_margin, 251 - warp_y_margin
pts2_x, pts2_y = 20 - warp_x_margin, 480 + warp_y_margin
pts3_x, pts3_y = 440 + warp_x_margin, 251 - warp_y_margin
pts4_x, pts4_y = 620 + warp_x_margin, 480 + warp_y_margin

warp_src = np.array([
    [pts1_x, pts1_y],
    [pts2_x, pts2_y],
    [pts3_x, pts3_y],
    [pts4_x, pts4_y],
], dtype = np.float32)

warp_dist = np.array([
[0                         ,                 0],
[(warp_image_width / 12) * 5, warp_image_height],
[warp_image_width          ,                 0],
[(warp_image_width / 12) * 7, warp_image_height],
], dtype = np.float32)

image_width, image_height = 640, 480

capture = cv2.VideoCapture(3)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capture.set(cv2.CAP_PROP_EXPOSURE, -10000)

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


def calibrate_image(src):
    global image_width, image_height
    dst = cv2.remap(src, map1, map2, cv2.INTER_LINEAR)
    dst = dst[y : y + h, x : x + w]
    return cv2.resize(dst, (image_width, image_height))

def warp_image(img, src, dst, size):
    src_to_dst_mtx = cv2.getPerspectiveTransform(src, dst)
    dst_to_src_mtx = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, src_to_dst_mtx, size, flags = cv2.INTER_LINEAR)
    return warp_img, src_to_dst_mtx, dst_to_src_mtx

def draw_circle(src, x, y):
    cv2.circle(src, (x, y), 2 , (255, 0, 0), -1)

cal = cv2.imread("/home/nvidia/xycar_ws/src/ObjectDetectionSystem/src/ObjectDetectionSystem/mask_image.png")
cal = calibrate_image(cal)
cv2.imwrite("/home/nvidia/xycar_ws/src/image_calibration_test/src/cal_mask.png", cal)

# while capture.isOpened():
    # _, src = capture.read()
    # cal = calibrate_image(src)
    # warp, src_mtx, dst_mtx = warp_image(cal, warp_src, warp_dist, (warp_image_width, warp_image_height))
    # #cv2.circle(cal, (pts1_x, pts1_y), 20, (255, 0, 0), -1)
    # #cv2.circle(cal, (pts2_x, pts2_y), 20, (255, 0, 0), -1)
    # #cv2.circle(cal, (pts3_x, pts3_y), 20, (255, 0, 0), -1)
    # #cv2.circle(cal, (pts4_x, pts4_y), 20, (255, 0, 0), -1)
    # # Raw
    # # 0
    # draw_circle(cal, 214, 253)  # 0, 0
    # draw_circle(cal, 251, 253)  # 0, 1
    # draw_circle(cal, 288, 254)  # 0, 2
    # draw_circle(cal, 328, 254)  # 0, 3
    # draw_circle(cal, 368, 255)  # 0, 4
    # draw_circle(cal, 408, 254)  # 0, 5

    # # 1
    # draw_circle(cal, 237, 256)  # 1, 1
    # draw_circle(cal, 281, 257)  # 1, 2
    # draw_circle(cal, 327, 259)  # 1, 3
    # draw_circle(cal, 377, 258)  # 1, 4
    # draw_circle(cal, 423, 258)  # 1, 5

    # # 2
    # draw_circle(cal, 216, 260)  # 2, 1
    # draw_circle(cal, 272, 262)  # 2, 2
    # draw_circle(cal, 328, 262)  # 2, 3
    # draw_circle(cal, 389, 263)  # 2, 4
    # draw_circle(cal, 445, 263)  # 2, 5

    # # 3
    # draw_circle(cal, 256, 269)  # 3, 2
    # draw_circle(cal, 328, 270)  # 3, 3
    # draw_circle(cal, 408, 270)  # 3, 4

    # # 4
    # draw_circle(cal, 225, 281)  # 4, 2
    # draw_circle(cal, 329, 286)  # 4, 3
    # draw_circle(cal, 441, 284)  # 4, 4

    # # 5
    # draw_circle(cal, 148, 315)  # 5, 2
    # draw_circle(cal, 329, 323)  # 5, 3
    # draw_circle(cal, 526, 318)  # 5, 4
    # # cv2.imshow('src', src)

    # cv2.imshow('calibrate image', cal)
    # cv2.imshow('warp_interlinear', warp)


