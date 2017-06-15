import cv2
import numpy as np
import math
import time
import thread


class HandReco:
    __blurX = 35
    __blurY = 35
    __threshValue = 127
    __kernelPixel = 6
    __startTime = time.time()
    __bg_captured = 0
    __count = 0
    __lastStatus = 'nothing'
    __lastFlip = __lastlastFlip = 'handfist'

    currentGesture = 'nothing'

    def __init__(self):
        cv2.namedWindow('control panel')
        cv2.createTrackbar('blur', 'control panel', 0, 50, self.__set_blur_value)
        cv2.createTrackbar('thresh', 'control panel', 0, 255, self.__set_thresh_value)
        cv2.createTrackbar('kernelPixel', 'control panel', 0, 8, self.__set_kernel_pixel)

        cv2.setTrackbarPos('blur', 'control panel', self.__blurX)
        cv2.setTrackbarPos('thresh', 'control panel', self.__threshValue)
        cv2.setTrackbarPos('kernelPixel', 'control panel', self.__kernelPixel)

    # def __init__(self, blur_value, thresh_value, kernel_pixel):
    #     self.blurX = self.blurY = blur_value
    #     self.threshValue = thresh_value
    #     self.kernelPixel = kernel_pixel

    def __post_to_server(self, gesture):
        if time.time() - self.__startTime > 0.3 and gesture != self.__lastStatus:
            self.__startTime = time.time()
            self.__lastStatus = gesture
            print str(time.time()) + ' posting ' + gesture
            self.currentGesture = gesture

    def __data_stream_filter(self, count_defects, crop_img):
        if count_defects > 3:
            status = 'handopen'
        else:
            status = 'handfist'

        if self.__lastFlip == status and self.__lastlastFlip == status:
            # self.post_to_server(status)
            pass

        cv2.putText(crop_img, self.currentGesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

        self.__lastlastFlip = self.__lastFlip
        self.__lastFlip = status

    def __remove_bg(self, frame):
        fg_mask = self.bg_model.apply(frame)
        kernel = np.ones((self.__kernelPixel, self.__kernelPixel), np.uint8)
        fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
        frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
        frame[fg_mask == 0] = (255, 255, 255)
        return frame

    def __nothing(self, x):
        pass

    def __set_blur_value(self, x):
        self.__blurX = self.__blurY = x if x % 2 == 1 else x + 1

    def __set_thresh_value(self, x):
        self.__threshValue = x

    def __set_kernel_pixel(self, x):
        self.__kernelPixel = x

    def _wrap(self):
        # choose camera
        if cv2.VideoCapture(1).isOpened():
            cap = cv2.VideoCapture(1)
        else:
            cap = cv2.VideoCapture(0)

        count = 1
        while cap.isOpened():
            ret, img = cap.read()
            print count
            count += 1
            if ret:
                img = cv2.flip(img, 1)
                # create rectangle area
                cv2.rectangle(img, (640, 480), (320, 0), (0, 255, 0), 0)
                # crop_img = img
                crop_img = img[0:480, 320:640]
                # cv2.imshow('raw', img)
                if self.__bg_captured:
                    crop_img = self.__remove_bg(crop_img)
                    # cv2.imshow('front', crop_img)

                grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

                blurred = cv2.GaussianBlur(grey, (self.__blurX, self.__blurY), 0)
                # _, thresh1 = cv2.threshold(blurred, threshValue, 255, cv2.THRESH_BINARY)
                _, thresh1 = cv2.threshold(blurred, self.__threshValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # cv2.imshow('Thresholded', thresh1)

                (version, _, _) = cv2.__version__.split('.')

                if version is '3':
                    image, contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                elif version is '2':
                    contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                try:
                    cnt = max(contours, key=lambda x: cv2.contourArea(x))
                except Exception, e:
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
                self.__data_stream_filter(count_defects, crop_img)
                # cv2.imshow('drawing', drawing)
                # cv2.imshow('end', crop_img)
                cv2.imshow('black', drawing)
                all_img = np.hstack((drawing, crop_img))
                cv2.imshow('Contours', all_img)
                interrupt = cv2.waitKey(10)
                if interrupt & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
                elif interrupt & 0xFF == ord('b'):
                    self.bg_model = cv2.BackgroundSubtractorMOG2(0, 10)
                    self.__bg_captured = 1

    def start(self):
        thread.start_new_thread(self._wrap, ())