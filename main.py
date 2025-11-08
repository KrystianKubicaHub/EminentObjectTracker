import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def wybranieObiektu():
    _, frame = videoCapture.read()
    r = cv.selectROI("ROI", frame);
    xTopLeft,yTopLeft,w,h = map(int,r)
    cv.destroyWindow("ROI")
    return xTopLeft,yTopLeft,w,h




def sledzenie(x, y, w, h, videoCapture):

    _, frame = videoCapture.read()
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.show()
    #cv.imshow('klatka1',frame)
    # x = 785
    # y = 470
    # w = 89
    # h = 119

    roi = frame[y:y + h, x:x + w]
    kwadrat = (x, y, w, h)

    cv.imshow('fura',roi)

    hsvRoi = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
    dolnyLimit = np.array([0, 0, 0])
    gornyLimit = np.array([179, 255, 255])

    maska = cv.inRange(hsvRoi,dolnyLimit,gornyLimit)
    histogramRoi = cv.calcHist([hsvRoi],[0],maska,[180],[0,180])
    cv.normalize(histogramRoi,histogramRoi,0,255,cv.NORM_MINMAX)
    odciecie = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,10,1)
    kolor = (255,0,0)

    #cv.imshow('hsv obrazek', cv.cvtColor(frame,cv.COLOR_BGR2HSV))

    #cv.imshow('hsv',hsvRoi)



    while True:
        xd ,frame = videoCapture.read()
        if xd == True:
            hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
            backProject = cv.calcBackProject([hsv],[0],histogramRoi,[0,180],1)

            ret,kwadrat = cv.CamShift(backProject, kwadrat, odciecie)
            box = cv.boxPoints(ret)

            #print(box)
            frame = cv.polylines(frame, [np.int32(box)], True, kolor, 3)

            #cv.imshow('backProj', backProject)
            #cv.imshow('mask', maska)

            cv.imshow('fura',frame)
            cv.waitKey(15)



if __name__ == "__main__":
    root = os.getcwd()
    video_path = os.path.join(root, 'videa', 'furaNaWsi.mp4')
    videoCapture = cv.VideoCapture(video_path)

    x,y,w,h = wybranieObiektu()

    sledzenie(x,y,w,h, videoCapture)