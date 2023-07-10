import cv2

cap = cv2.VideoCapture(3)

while cap.isOpened():
    ret, frame = cap.read()
    H, L, S = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HLS))

    g_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("gray", g_frame)
    cv2.imshow("H", H)
    cv2.imshow("L", L)
    cv2.imshow("S", S)

    if cv2.waitKey(10) == 27:
        cap.release()
        break
