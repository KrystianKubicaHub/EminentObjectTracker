import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import filedialog

WINDOW_NAME = "tracked"
MAX_WIDTH = 1600

MODELE = {
    "1": "CamShift",
    "2": "MeanShift"
}

def resize_frame(frame, max_width=MAX_WIDTH):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame

    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)


def wyborModelu():
    print("\nDostępne modele śledzenia:")
    print("-------------------------")
    for k, v in MODELE.items():
        print(f"{k}. {v}")
    print("-------------------------")

    wybor = input("Wpisz numer wybranego modelu: ").strip()

    if wybor not in MODELE:
        print("Niepoprawny wybór modelu.")
        exit()

    return MODELE[wybor]


def wybranieObiektu():
    boolean, frame = videoCapture.read()
    if not boolean:
        print("Nie można wczytać wideo")
        exit()

    frame = resize_frame(frame)

    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    cv.resizeWindow(WINDOW_NAME, frame.shape[1], frame.shape[0])

    region = cv.selectROI(WINDOW_NAME, frame, fromCenter=False, showCrosshair=True)
    xTopLeft, yTopLeft, w, h = map(int, region)

    cv.destroyWindow(WINDOW_NAME)
    return xTopLeft, yTopLeft, w, h


def sledzenie(x, y, w, h, videoCapture, model):

    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)

    boolean, frame = videoCapture.read()
    if not boolean:
        print("Nie można wczytać wideo")
        exit()

    frame = resize_frame(frame)
    cv.resizeWindow(WINDOW_NAME, frame.shape[1], frame.shape[0])

    roi = frame[y:y + h, x:x + w]
    kwadrat = (x, y, w, h)

    hsvRoi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    h_channel, s_channel, v_channel = cv.split(hsvRoi)

    h_mean, h_std = np.mean(h_channel), np.std(h_channel)
    s_mean, s_std = np.mean(s_channel), np.std(s_channel)
    v_mean, v_std = np.mean(v_channel), np.std(v_channel)

    tolerance = 1.5

    h_min, h_max = max(0, h_mean - tolerance * h_std), min(179, h_mean + tolerance * h_std)
    s_min, s_max = max(0, s_mean - tolerance * s_std), min(255, s_mean + tolerance * s_std)
    v_min, v_max = max(0, v_mean - tolerance * v_std), min(255, v_mean + tolerance * v_std)

    dolnyLimit = np.array([h_min, s_min, v_min])
    gornyLimit = np.array([h_max, s_max, v_max])

    maska = cv.inRange(hsvRoi, dolnyLimit, gornyLimit)
    histogramRoi = cv.calcHist([hsvRoi], [0], maska, [180], [0, 180])
    cv.normalize(histogramRoi, histogramRoi, 0, 255, cv.NORM_MINMAX)

    odciecie = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    kolor = (255, 0, 0)

    while True:
        ret, frame = videoCapture.read()
        if not ret:
            break

        frame = resize_frame(frame)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        backProject = cv.calcBackProject([hsv], [0], histogramRoi, [0, 180], 1)

        if model == "CamShift":
            ret_box, kwadrat = cv.CamShift(backProject, kwadrat, odciecie)
            box = cv.boxPoints(ret_box)
            frame = cv.polylines(frame, [np.int32(box)], True, kolor, 3)

        elif model == "MeanShift":
            _, kwadrat = cv.meanShift(backProject, kwadrat, odciecie)
            x, y, w, h = kwadrat
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), kolor, 3)

        cv.imshow(WINDOW_NAME, frame)
        if cv.waitKey(15) & 0xFF == 27:
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(script_dir, "videa")

    tk_root = tk.Tk()
    tk_root.withdraw()

    video_path = filedialog.askopenfilename(
        title="Wybierz plik wideo do analizy",
        initialdir=video_dir,
        filetypes=[("Pliki wideo", "*.mp4")]
    )

    if not video_path:
        print("Nie wybrano pliku wideo.")
        exit()

    videoCapture = cv.VideoCapture(video_path)
    if not videoCapture.isOpened():
        print("Nie można otworzyć wybranego wideo.")
        exit()

    model = wyborModelu()
    x, y, w, h = wybranieObiektu()
    sledzenie(x, y, w, h, videoCapture, model)
