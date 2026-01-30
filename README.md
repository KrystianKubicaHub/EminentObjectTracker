# Eminent Object Tracker

## What is this thing?

This project emerged from the chaotic depths of a university assignment. Its primary objective? To fabricate a sophisticated application that facilitates the comparison of diverse tracking algorithms for object tracking in video sequences. 

The application enables testing of multiple tracking methodologies, with some supporting advanced color space transformations for enhanced performance. You can draw a rectangle on your video like a kindergartner with a crayon, and then watch the magic happen.

### Supported Tracking Models

**Histogram-Based Trackers** (support color space selection):
- **CamShift** - Color histogram with rotation tracking
  - Supports: HSV, RGB, YCbCr, LAB color spaces
- **MeanShift** - Color histogram tracking without rotation
  - Supports: HSV, RGB, YCbCr, LAB color spaces

**Correlation Filter Trackers**:
- **CSRT** - Discriminative Correlation Filter with Spatial Reliability
- **KCF** - Kernelized Correlation Filter
- **MOSSE** - Minimum Output Sum of Squared Error (fast but basic)
- **MIL** - Multiple Instance Learning tracker

**Deep Learning Trackers**:
- **YOLOv8** - State-of-the-art object detection with ByteTrack
  - Automatically downloads model on first run
  - Supports persistent tracking across frames

### Features That Actually Work

- **Color Space Selection** - For CamShift and MeanShift (HSV, RGB, YCbCr, LAB)
- **Interactive ROI Selection** - Draw rectangle directly on video frame
- **Pause/Resume** - Because sometimes you need a break
- **Trajectory Traces** - Visualize object path with customizable color and thickness
- **Cancel Button** - When everything goes to shit and you want to start over
- **Dark Theme** - Your eyes will thank you
- **Video Thumbnail Preview** - See what you're working with

## How to Run This Beast

### Installation

First, install all the dependencies. This might take a while, so grab a coffee or contemplate your life choices:

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

After installation is complete and your existential crisis has subsided:

```bash
python3 main.py
```

### Requirements

- Python 3.9 or higher
- macOS / Linux / Windows (tested on macOS M4 Pro)
- Webcam or video files for testing
- Patience (especially with YOLO model download on first run)

---

**Note**: This is a student project. Expect bugs, weird behavior, and occasional moments of brilliance. If something breaks, that's a feature, not a bug.
