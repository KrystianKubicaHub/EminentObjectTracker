import os
import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import filedialog


COLOR_SPACES = {
    "1": {
        "name": "HSV",
        "convert": cv.COLOR_BGR2HSV,
        "channels": [0],
        "ranges": [0, 180],
        "bins": [180]
    },
    "2": {
        "name": "RGB",
        "convert": cv.COLOR_BGR2RGB,
        "channels": [0, 1, 2],
        "ranges": [0, 256, 0, 256, 0, 256],
        "bins": [8, 8, 8]
    },
    "3": {
        "name": "YCbCr",
        "convert": cv.COLOR_BGR2YCrCb,
        "channels": [1, 2],
        "ranges": [0, 256, 0, 256],
        "bins": [32, 32]
    },
    "4": {
        "name": "LAB",
        "convert": cv.COLOR_BGR2LAB,
        "channels": [1, 2],
        "ranges": [0, 256, 0, 256],
        "bins": [32, 32]
    }
}

MODELE = {
    "1": "CamShift",
    "2": "MeanShift"
}

WINDOW_NAME = "tracked"
MAX_WIDTH = 1600


def resize_frame(frame, max_width=MAX_WIDTH):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)


def wyborModelu():
    print("\nDostępne modele śledzenia:")
    print("-------------------------")
    for k, v in MODELE.items():
        print(f"{k}. {v}")
    print("-------------------------")

    wybor = input("Wpisz numer modelu: ").strip()
    if wybor not in MODELE:
        print("Niepoprawny wybór modelu.")
        exit()
    return MODELE[wybor]


def wyborPrzestrzeniBarw():
    print("\nDostępne przestrzenie barw:")
    print("-------------------------")
    for k, v in COLOR_SPACES.items():
        print(f"{k}. {v['name']}")
    print("-------------------------")

    wybor = input("Wpisz numer przestrzeni barw: ").strip()
    if wybor not in COLOR_SPACES:
        print("Niepoprawny wybór przestrzeni barw.")
        exit()
    return COLOR_SPACES[wybor]


def wybranieObiektu(videoCapture):
    ret, frame = videoCapture.read()
    if not ret:
        print("Nie można wczytać wideo")
        exit()

    frame = resize_frame(frame)

    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    cv.resizeWindow(WINDOW_NAME, frame.shape[1], frame.shape[0])

    x, y, w, h = map(
        int,
        cv.selectROI(
            WINDOW_NAME,
            frame,
            fromCenter=False,
            showCrosshair=True
        )
    )

    cv.destroyWindow(WINDOW_NAME)
    return x, y, w, h


def sledzenie(x, y, w, h, videoCapture, model, color_cfg):

    ret, frame = videoCapture.read()
    if not ret:
        print("Nie można wczytać wideo")
        exit()

    frame = resize_frame(frame)
    roi = frame[y:y + h, x:x + w]
    track_window = (x, y, w, h)

    # punkt początkowy (środek ROI)
    start_center = [x + w // 2, y + h // 2]

    roi_cs = cv.cvtColor(roi, color_cfg["convert"])

    histogramRoi = cv.calcHist(
        [roi_cs],
        color_cfg["channels"],
        None,
        color_cfg["bins"],
        color_cfg["ranges"]
    )
    cv.normalize(histogramRoi, histogramRoi, 0, 255, cv.NORM_MINMAX)

    term_criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    kolor = (255, 0, 0)
    tekst_kolor = (0, 0, 0)  # czarny

    while True:
        ret, frame = videoCapture.read()
        if not ret:
            break

        frame = resize_frame(frame)
        frame_cs = cv.cvtColor(frame, color_cfg["convert"])

        backProject = cv.calcBackProject(
            [frame_cs],
            color_cfg["channels"],
            histogramRoi,
            color_cfg["ranges"],
            1
        )

        if model == "CamShift":
            ret_box, track_window = cv.CamShift(backProject, track_window, term_criteria)
            box = cv.boxPoints(ret_box)
            frame = cv.polylines(frame, [np.int32(box)], True, kolor, 3)

            cx = int(ret_box[0][0])
            cy = int(ret_box[0][1])

        elif model == "MeanShift":
            _, track_window = cv.meanShift(backProject, track_window, term_criteria)
            x, y, w, h = track_window
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), kolor, 3)

            cx = x + w // 2
            cy = y + h // 2

        # wektor przesunięcia względem pozycji początkowej
        dx = cx - start_center[0]
        dy = cy - start_center[1]

        # rysowanie wektora
        cv.arrowedLine(
            frame,
            start_center,
            (cx, cy),
            (0, 0, 255),
            2,
            tipLength=0.2
        )

        cv.putText(
            frame,
            f"Model: {model} | Color: {color_cfg['name']}",
            (20, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.9,
            tekst_kolor,
            2
        )

        cv.putText(
            frame,
            f"dx={dx}, dy={dy}",
            (20, 65),
            cv.FONT_HERSHEY_SIMPLEX,
            0.9,
            tekst_kolor,
            2
        )

        start_center[0] = cx
        start_center[1] = cy

        cv.imshow(WINDOW_NAME, frame)
        if cv.waitKey(15) & 0xFF == 27:
            break

    videoCapture.release()
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
        print("Nie można otworzyć wideo.")
        exit()

    model = wyborModelu()
    color_cfg = wyborPrzestrzeniBarw()
    x, y, w, h = wybranieObiektu(videoCapture)
    sledzenie(x, y, w, h, videoCapture, model, color_cfg)
