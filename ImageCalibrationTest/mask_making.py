import numpy as np
import cv2

cap = cv2.VideoCapture(0)

mask_image = np.zeros([480, 640, 1], dtype = np.uint8)
mask_image.fill(255)

while (1):
    ret, frame = cap.read() 
    frame = cv2.resize(frame,(640,480))
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # detect white
    _, L, _ = cv2.split(cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HLS))

    _, binary_frame = cv2.threshold(L,180,255,cv2.THRESH_BINARY_INV)

    mask_image = cv2.min(binary_frame, mask_image)

    cv2.imshow('original',frame)
    cv2.imshow('frame',binary_frame)
    cv2.imshow("mask_image", mask_image)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        cv2.imwrite('/home/nvidia/xycar_ws/src/image_calibration_test/src/mask_image.png', mask_image)
        break
        
cap.release()
cv2.destroyAllWindows()