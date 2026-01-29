import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def wybranieObiektu():
    boolean, frame = videoCapture.read()
    if boolean == False:
        print("Nie można wczytać wideo")
        exit()
    region = cv.selectROI("Select your region of interest", frame)
    xTopLeft, yTopLeft, w, h = map(int,region)
    cv.destroyWindow("ROI")
    return xTopLeft,yTopLeft,w,h




def sledzenie(x, y, w, h, videoCapture):
    
    boolean, frame = videoCapture.read()
    if boolean == False:
        print("Nie można wczytać wideo")
        exit()
    #frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #plt.imshow(frame_rgb)
    #plt.show()

    roi = frame[y:y + h, x:x + w]
    kwadrat = (x, y, w, h)

    cv.imshow('fura',roi)

    hsvRoi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    
    h_channel, s_channel, v_channel = cv.split(hsvRoi)
    
    h_mean = np.mean(h_channel)
    h_std = np.std(h_channel)
    s_mean = np.mean(s_channel)
    s_std = np.std(s_channel)
    v_mean = np.mean(v_channel)
    v_std = np.std(v_channel)
    
    tolerance = 1.5
    
    h_min = max(0, h_mean - tolerance * h_std)
    h_max = min(179, h_mean + tolerance * h_std)
    s_min = max(0, s_mean - tolerance * s_std)
    s_max = min(255, s_mean + tolerance * s_std)
    v_min = max(0, v_mean - tolerance * v_std)
    v_max = min(255, v_mean + tolerance * v_std)
    
    dolnyLimit = np.array([h_min, s_min, v_min])
    gornyLimit = np.array([h_max, s_max, v_max])
    
    print(f" Dynamicznie dobrane limity HSV:")
    print(f"   Hue:        {h_min:.0f} - {h_max:.0f}  (średnia: {h_mean:.0f})")
    print(f"   Saturation: {s_min:.0f} - {s_max:.0f}  (średnia: {s_mean:.0f})")
    print(f"   Value:      {v_min:.0f} - {v_max:.0f}  (średnia: {v_mean:.0f})")

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
        else:
            break


if __name__ == "__main__":
    root = os.getcwd() #example root /Users/krystian.kubica/AndroidStudioProjects/EminentObjectTracker
   # video_path = os.path.join(root, 'videa', 'furaNaWsi.mp4')
    video_path = os.path.join(root, 'videa', '4062991-uhd_3840_2160_30fps.mp4')
   # video_path = os.path.join(root, 'videa', '5845159-uhd_3840_2160_30fps.mp4')
    videoCapture = cv.VideoCapture(video_path)

    x,y,w,h = wybranieObiektu()

    sledzenie(x,y,w,h, videoCapture)