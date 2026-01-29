import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2 as cv

matplotlib.use('TkAgg')

# =====================================================
# Konfiguracja przestrzeni barw
# =====================================================

COLOR_SPACES = {
    "HSV": {
        "convert": cv.COLOR_BGR2HSV,
        "channels": [0],            # Hue
        "ranges": [0, 180],
        "bins": [180]
    },
    "RGB": {
        "convert": cv.COLOR_BGR2RGB,
        "channels": [0, 1, 2],      # R, G, B
        "ranges": [0, 256, 0, 256, 0, 256],
        "bins": [8, 8, 8]
    },
    "YCbCr": {
        "convert": cv.COLOR_BGR2YCrCb,
        "channels": [1, 2],         # Cb, Cr
        "ranges": [0, 256, 0, 256],
        "bins": [32, 32]
    },
    "LAB": {
        "convert": cv.COLOR_BGR2LAB,
        "channels": [1, 2],     # a, b
        "ranges": [0, 256, 0, 256],
        "bins": [32, 32]
    }

}
import tkinter as tk
from tkinter import filedialog

# =====================================================
# Wybór obiektu (ROI)
# =====================================================

def wybranieObiektu(videoCapture):
    _, frame = videoCapture.read()
    if _ == False:
        print("Nie można wczytać wideo")
        exit()
    r = cv.selectROI("Select your region of interest", frame)
    x, y, w, h = map(int, r)
    return x, y, w, h


# =====================================================
# Śledzenie obiektu (CamShift)
# =====================================================

def sledzenie(x, y, w, h, videoCapture, color_space="HSV"):

    cfg = COLOR_SPACES[color_space]

    _, frame = videoCapture.read()
    if _ == False:
        print("Nie można wczytać wideo")
        exit()
    roi = frame[y:y + h, x:x + w]
    track_window = (x, y, w, h)

    # --- ROI w wybranej przestrzeni barw ---
    roi_cs = cv.cvtColor(roi, cfg["convert"])

    # Histogram ROI
    histogramRoi = cv.calcHist(
        [roi_cs],
        cfg["channels"],
        None,
        cfg["bins"],
        cfg["ranges"]
    )

    cv.normalize(histogramRoi, histogramRoi, 0, 255, cv.NORM_MINMAX)

    # Kryteria CamShift
    term_criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

    kolor = (255, 0, 0)

    while True:
        ret, frame = videoCapture.read()
        if not ret:
            break

        frame_cs = cv.cvtColor(frame, cfg["convert"])

        # Backprojection
        backProject = cv.calcBackProject(
            [frame_cs],
            cfg["channels"],
            histogramRoi,
            cfg["ranges"],
            1
        )

        ret, track_window = cv.CamShift(backProject, track_window, term_criteria)

        box = cv.boxPoints(ret)
        box = np.int32(box)

        frame = cv.polylines(frame, [box], True, kolor, 3)

        cv.putText(
            frame,
            f"Color space: {color_space}",
            (20, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv.imshow("sledzenie", frame)

        if cv.waitKey(15) & 0xFF == 27:  # ESC
            break

    videoCapture.release()
    cv.destroyAllWindows()


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
##############################
    root = os.getcwd()
    video_path = os.path.join(root, "videa", "furaNaWsi.mp4")

    # katalog, w którym leży main.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(script_dir, "videa")

    # --- GUI do wyboru pliku ---
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


    x, y, w, h = wybranieObiektu(videoCapture)

    if not videoCapture.isOpened():
        print("Nie można otworzyć wybranego wideo.")
        exit()

    # Wybierz: "HSV", "RGB", "YCbCr"
    sledzenie(x, y, w, h, videoCapture, color_space="YCbCr")
