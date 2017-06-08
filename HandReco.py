import cv2
import numpy as np
import math

def remove_bg(frame):
    fg_mask=bg_model.apply(frame)
    kernel = np.ones((kernelPixel,kernelPixel),np.uint8)
    fg_mask=cv2.erode(fg_mask,kernel,iterations = 1)
    frame=cv2.bitwise_and(frame,frame,mask=fg_mask)
    frame[fg_mask == 0] = (255, 255, 255)
    return frame

def nothing(x):
    pass

def create_control_panel():
    cv2.namedWindow('control panel')
    cv2.createTrackbar('blur', 'control panel', 0, 50, setBlurValue)
    cv2.createTrackbar('thresh', 'control panel', 0, 255, setThreshValue)
    cv2.createTrackbar('kernelPixel', 'control panel', 0, 8, setKernelPixel)

def setBlurValue(x):
    global blurX, blurY
    blurX = blurY = x if x % 2 == 1 else x + 1

def setThreshValue(x):
    global threshValue
    threshValue = x

def setKernelPixel(x):
    global kernelPixel
    kernelPixel = x

def initParameter():
    global blurX, blurY, threshValue, kernelPixel
    blurX = 35
    blurY = 35
    cv2.setTrackbarPos('blur', 'control panel', 35)
    threshValue = 127
    cv2.setTrackbarPos('thresh', 'control panel', 127)
    kernelPixel = 5
    cv2.setTrackbarPos('kernelPixel', 'control panel', 5)

# class HandReco:
#     def currentGesture(self):
#         global currentGesture
#         return currentGesture


# ----------parameters in control panel-------------
global blurX, blurY, threshValue, kernelPixel
# --------------------------------------------------
global currentGesture
currentGesture = 'haha'

create_control_panel()
initParameter()
cap = cv2.VideoCapture(1)
bg_captured = 0

while (cap.isOpened()):
    fo = open("gesture", "wb")
    ret, img = cap.read()
    if ret:
        img = cv2.flip(img, 1)
        # create rectangle area
        cv2.rectangle(img, (640, 480), (320, 0), (0, 255, 0), 0)
        # crop_img = img
        crop_img = img[0:480, 320:640]
        cv2.imshow('raw', img)
        if(bg_captured):
            crop_img = remove_bg(crop_img)
            cv2.imshow('front', crop_img)

        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(grey, (blurX, blurY), 0)
        # _, thresh1 = cv2.threshold(blurred, threshValue, 255, cv2.THRESH_BINARY)
        _, thresh1 = cv2.threshold(blurred, threshValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        cv2.imshow('Thresholded', thresh1)

        (version, _, _) = cv2.__version__.split('.')

        if version is '3':
            image, contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        elif version is '2':
            contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        try:
            cnt = max(contours, key=lambda x: cv2.contourArea(x))
        except Exception,e:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)
        hull = cv2.convexHull(cnt)
        drawing = np.zeros(crop_img.shape, np.uint8)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        count_defects = 0
        cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)
        if defects is None:
            continue
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_img, far, 1, [0, 0, 255], -1)
            # dist = cv2.pointPolygonTest(cnt,far,True)
            cv2.line(crop_img, start, end, [0, 255, 0], 2)
            # cv2.circle(crop_img,far,5,[0,0,255],-1)
        if count_defects > 2:
            cv2.putText(crop_img, "handopen", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            fo.write('handopen')
        else:
            cv2.putText(crop_img, "handfist", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            fo.write('handfist')
        # cv2.imshow('drawing', drawing)
        # cv2.imshow('end', crop_img)
        all_img = np.hstack((drawing, crop_img))
        cv2.imshow('Contours', all_img)
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            fo.close()
            break
        elif interrupt & 0xFF == ord('b'):
            bg_model = cv2.BackgroundSubtractorMOG2(0, 10)
            bg_captured = 1
