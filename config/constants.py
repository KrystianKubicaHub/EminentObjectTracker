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

class Sizes:
    WINDOW_WIDTH = 1400
    WINDOW_HEIGHT = 900
    VIDEO_MAX_WIDTH = 1600
    SIDEBAR_WIDTH = 350

class Models:
    OPENCV_TRACKERS = {
        "CamShift": {"supports_color_space": True, "description": "chuj "},
        "CSRT": {"supports_color_space": True, "description": "chuj2 "},
        "KCF": {"supports_color_space": True, "description": "chuj3 "},
        "MOSSE": {"supports_color_space": False, "description": "chuj4 "},
        "MIL": {"supports_color_space": False, "description": "chuj5 "},
    }
    
    DEEP_TRACKERS = {
        "YOLOv8": {"supports_color_space": False, "description": "chuj "},
    }

class ColorSpaces:
    AVAILABLE = {
        "HSV": {"convert": "COLOR_BGR2HSV", "channels": [0], "ranges": [0, 180], "bins": [180]},
        "RGB": {"convert": "COLOR_BGR2RGB", "channels": [0, 1, 2], "ranges": [0, 256, 0, 256, 0, 256], "bins": [8, 8, 8]},
        "YCbCr": {"convert": "COLOR_BGR2YCrCb", "channels": [1, 2], "ranges": [0, 256, 0, 256], "bins": [32, 32]},
        "LAB": {"convert": "COLOR_BGR2LAB", "channels": [1, 2], "ranges": [0, 256, 0, 256], "bins": [32, 32]},
    }

class Strings:
    APP_TITLE = "Eminent Object Tracker"
    SELECT_VIDEO = "Select videło"
    SELECT_MODEL = "select your Boom Boom"
    SELECT_COLOR_SPACE = "Select Color Space"
    ENABLE_TRACE = " enable trace or not enable trace"
    START_TRACKING = "Jazda z kurwami"
    SELECT_ROI = "Zaznacz kwadrat"
    TRACKING_ACTIVE = "LEcimy"
    NO_VIDEO = "No videło"
    ERROR = "Chuj i chuj"
