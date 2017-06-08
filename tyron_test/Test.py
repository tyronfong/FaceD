import numpy as np
import cv2

def remove_bg(frame):
    fg_mask=bg_model.apply(frame)
    kernel = np.ones((3,3),np.uint8)
    fg_mask=cv2.erode(fg_mask,kernel,iterations = 1)
    frame=cv2.bitwise_and(frame,frame,mask=fg_mask)
    return frame

cap = cv2.VideoCapture(1)
bg_captured = 0

while (cap.isOpened()):
    ret, img = cap.read()
    if ret:
        cv2.imshow('tyron', img)
        if(bg_captured):
            frontImg = remove_bg(img)
            cv2.imshow('frontImg', frontImg)


    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('q'):
        break
    elif interrupt & 0xFF == ord('b'):
        bg_model = cv2.BackgroundSubtractorMOG2(0, 10)
        bg_captured = 1