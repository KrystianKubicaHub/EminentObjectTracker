# Eminent Object Tracker

## What is it?

This project was created under chaotic circumstances during a university assignment. Its main goal was to create a military-space-scientific-genius-project-clever application that would facilitate the comparison of various object tracking algorithms in video sequences. 

The application allows you to test multiple tracking methods, and for models using different colour spaces, you can also change the colour spaces. Anka also implements drawing the path of an object tracked in a static video.

### Supported tracking models

**Histogram-based tracking** (supports colour space selection):
- **CamShift** - colour histogram with rotation tracking
  - Supports: HSV, RGB, YCbCr, LAB colour spaces
- **MeanShift** - colour histogram tracking without rotation
  — Supports: HSV, RGB, YCbCr, LAB colour spaces

**Correlation filter-based tracking**:
- **CSRT** — discriminative correlation filter with spatial reliability
- **KCF** — kernel correlation filter
- **MOSSE** — minimum sum of squared output errors (fast but basic)
- **MIL** — multiple learning tracker

**Deep learning trackers**:
- **YOLOv8** — state-of-the-art object detection with ByteTrack

## How to Run This Beast

### Installation

First, launch that amazing app.

```bash
pip3 install -r requirements.txt
```

The requirements include:
- `opencv-contrib-python>=4.8.0` - For all the OpenCV trackers
- `numpy>=1.24.0` - Because math
- `PySide6>=6.6.0` - Fancy GUI framework
- `ultralytics>=8.0.0` - YOLOv8 implementation
- `lap>=0.5.12` - Required for YOLO tracking

### Running the Application

After installation:

```bash
python3 main.py
```



**Note**: This is a student project. Expect bugs, weird behavior, and occasional moments of brilliance. If something breaks, that is a feature, not a bug.

We wish you very Good day
