from PySide6.QtGui import QColor

class Colors:
    BACKGROUND = QColor(18, 18, 18)
    SURFACE = QColor(28, 28, 28)
    PRIMARY = QColor(138, 43, 226)
    SECONDARY = QColor(75, 0, 130)
    ACCENT = QColor(186, 85, 211)
    TEXT = QColor(240, 240, 240)
    TEXT_SECONDARY = QColor(180, 180, 180)
    SUCCESS = QColor(0, 200, 83)
    WARNING = QColor(255, 152, 0)
    ERROR = QColor(244, 67, 54)
    TRACKER_BOX = (255, 0, 0)
    TRACE_LINE = (138, 43, 226)
    TRACE_THICKNESS = 2

class Sizes:
    WINDOW_WIDTH = 1400
    WINDOW_HEIGHT = 900
    VIDEO_MAX_WIDTH = 1600
    SIDEBAR_WIDTH = 350
    THUMBNAIL_HEIGHT = 200
    MAX_TRACE_POINTS = 5000

class Models:
    OPENCV_TRACKERS = {
        "CamShift": {"supports_color_space": True, "description": "Color histogram with rotation"},
        "MeanShift": {"supports_color_space": True, "description": "Color histogram tracking"},
        "CSRT": {"supports_color_space": False, "description": "Discriminative correlation filter"},
        "KCF": {"supports_color_space": False, "description": "Kernelized correlation filter"},
        "MOSSE": {"supports_color_space": False, "description": "Fast correlation filter"},
        "MIL": {"supports_color_space": False, "description": "Multiple instance learning"},
    }
    
    DEEP_TRACKERS = {
        "YOLOv8": {"supports_color_space": False, "description": "chuj "},
    }

class ColorSpaces:
    AVAILABLE = {
        "HSV": {"convert": "COLOR_BGR2HSV", "channels": [0], "ranges": [0, 180], "bins": [180]},
        "RGB": {"convert": "COLOR_BGR2RGB", "channels": [0, 1], "ranges": [0, 256, 0, 256], "bins": [32, 32]},
        "YCbCr": {"convert": "COLOR_BGR2YCrCb", "channels": [1, 2], "ranges": [0, 256, 0, 256], "bins": [32, 32]},
        "LAB": {"convert": "COLOR_BGR2LAB", "channels": [1, 2], "ranges": [0, 256, 0, 256], "bins": [32, 32]},
    }

class Strings:
    APP_TITLE = "Eminent Object Tracker"
    SELECT_VIDEO = "Select Video"
    SELECT_MODEL = "Select Tracking Algorithm"
    SELECT_COLOR_SPACE = "Select Color Space"
    ENABLE_TRACE = "Enable trajectory trace"
    TRACE_COLOR = "Trace Color"
    TRACE_THICKNESS = "Trace Thickness"
    START_TRACKING = "Start Tracking"
    STOP_TRACKING = "Pause"
    RESUME_TRACKING = "Resume"
    SELECT_ROI = "Draw rectangle on video to select object"
    TRACKING_ACTIVE = "Tracking in progress..."
    SELECT_VIDEO_FIRST = "← Select a video file to begin"
    ERROR = "Error"
    NO_VIDEO = "No video selected"
    BENCHMARK_MODE = "Benchmark All Models"
    BENCHMARK_RUNNING = "Benchmarking in progress..."
    BENCHMARK_COMPLETE = "Benchmark complete"
    EXPORT_RESULTS = "Export Results"
