import sys
import os
import cv2 as cv
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QComboBox, QLabel, 
                               QFileDialog, QCheckBox, QFrame, QMessageBox, 
                               QProgressBar, QScrollArea, QTextEdit, QTableWidget, 
                               QTableWidgetItem, QHeaderView)
from PySide6.QtCore import Qt, QTimer, Signal, QPoint, QRect, QThread
from PySide6.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QIcon, QColor
from config.constants import Colors, Sizes, Models, ColorSpaces, Strings
from utils.roi_selector import ROISelector
from utils.trace_drawer import TraceDrawer
from utils.video_processor import VideoProcessor
from utils.benchmark import BenchmarkEngine
from trackers.camshift_tracker import CamShiftTracker
from trackers.meanshift_tracker import MeanShiftTracker
from trackers.csrt_tracker import CSRTTracker
from trackers.kcf_tracker import KCFTracker
from trackers.mosse_tracker import MOSSETracker
from trackers.mil_tracker import MILTracker
from trackers.yolo_tracker import YOLOTracker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(Strings.APP_TITLE)
        
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "appIcon.png")
        if os.path.exists(icon_path):
            icon_img = cv.imread(icon_path, cv.IMREAD_UNCHANGED)
            if icon_img is not None:
                if icon_img.shape[2] == 3:
                    gray = cv.cvtColor(icon_img, cv.COLOR_BGR2GRAY)
                    _, alpha = cv.threshold(gray, 250, 255, cv.THRESH_BINARY_INV)
                    icon_img = cv.cvtColor(icon_img, cv.COLOR_BGR2BGRA)
                    icon_img[:, :, 3] = alpha
                
                temp_icon_path = os.path.join(os.path.dirname(icon_path), "temp_icon.png")
                cv.imwrite(temp_icon_path, icon_img)
                self.setWindowIcon(QIcon(temp_icon_path))
            else:
                self.setWindowIcon(QIcon(icon_path))
        
        self.resize(Sizes.WINDOW_WIDTH, Sizes.WINDOW_HEIGHT)
        self.showMaximized()
        
        self.video_path = None
        self.selected_model = None
        self.selected_color_space = None
        self.enable_trace = False
        self.tracker = None
        self.video_processor = None
        self.trace_drawer = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.roi_mode = False
        self.roi_start = None
        self.roi_end = None
        self.temp_frame = None
        self.is_tracking = False
        self.is_paused = False
        self.benchmark_mode = False
        self.benchmark_roi = None
        
        self.init_ui()
        self.apply_dark_theme()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.sidebar = self.create_sidebar()
        main_layout.addWidget(self.sidebar)
        
        self.video_area = QWidget()
        video_layout = QVBoxLayout(self.video_area)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(f"background-color: {Colors.BACKGROUND.name()}; border: none; font-size: 18px; color: {Colors.TEXT_SECONDARY.name()};")
        self.video_label.setText(Strings.SELECT_VIDEO_FIRST)
        self.video_label.setMouseTracking(True)
        video_layout.addWidget(self.video_label, 1)
        
        controls_layout = QHBoxLayout()
        controls_layout.addStretch()
        
        self.pause_resume_btn = self.create_button(Strings.STOP_TRACKING, self.toggle_pause)
        self.pause_resume_btn.setVisible(False)
        self.pause_resume_btn.setMaximumWidth(200)
        controls_layout.addWidget(self.pause_resume_btn)
        
        self.cancel_btn = self.create_button("Cancel", self.cancel_tracking)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.setMaximumWidth(200)
        self.cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ERROR.name()};
                color: {Colors.TEXT.name()};
                border: none;
                border-radius: 6px;
                padding: 12px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {Colors.WARNING.name()};
            }}
            QPushButton:pressed {{
                background-color: {Colors.ERROR.name()};
            }}
        """)
        controls_layout.addWidget(self.cancel_btn)
        
        self.reset_btn = self.create_button("Reset", self.reset_tracking)
        self.reset_btn.setVisible(False)
        self.reset_btn.setMaximumWidth(200)
        controls_layout.addWidget(self.reset_btn)
        
        controls_layout.addStretch()
        video_layout.addLayout(controls_layout)
        video_layout.addSpacing(20)
        
        main_layout.addWidget(self.video_area, 1)
    
    def create_sidebar(self):
        sidebar = QFrame()
        sidebar.setFixedWidth(Sizes.SIDEBAR_WIDTH)
        sidebar.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE.name()};
                border-right: 1px solid {Colors.PRIMARY.name()};
            }}
        """)
        
        self.sidebar_layout = QVBoxLayout(sidebar)
        self.sidebar_layout.setSpacing(20)
        self.sidebar_layout.setContentsMargins(20, 30, 20, 30)
        
        title = QLabel(Strings.APP_TITLE)
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet(f"color: {Colors.PRIMARY.name()}; border: none;")
        title.setAlignment(Qt.AlignCenter)
        self.sidebar_layout.addWidget(title)
        
        self.sidebar_layout.addSpacing(20)
        
        self.video_btn = self.create_button(Strings.SELECT_VIDEO, self.select_video)
        self.sidebar_layout.addWidget(self.video_btn)
        
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedHeight(Sizes.THUMBNAIL_HEIGHT)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setStyleSheet(f"""
            background-color: {Colors.BACKGROUND.name()};
            border: 2px solid {Colors.PRIMARY.name()};
            border-radius: 6px;
        """)
        self.thumbnail_label.hide()
        self.sidebar_layout.addWidget(self.thumbnail_label)
        
        self.video_info = QLabel("No video selected")
        self.video_info.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10px; border: none;")
        self.video_info.setWordWrap(True)
        self.sidebar_layout.addWidget(self.video_info)
        
        self.sidebar_layout.addSpacing(10)
        
        self.model_label = QLabel(Strings.SELECT_MODEL)
        self.model_label.setStyleSheet(f"color: {Colors.TEXT.name()}; font-weight: bold; border: none;")
        self.sidebar_layout.addWidget(self.model_label)
        
        self.model_combo = QComboBox()
        all_models = {**Models.OPENCV_TRACKERS, **Models.DEEP_TRACKERS}
        for model_name, info in all_models.items():
            self.model_combo.addItem(f"{model_name}", model_name)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        self.style_combo(self.model_combo)
        self.sidebar_layout.addWidget(self.model_combo)
        
        self.model_desc = QLabel()
        self.model_desc.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10px; border: none; font-style: italic;")
        self.model_desc.setWordWrap(True)
        self.sidebar_layout.addWidget(self.model_desc)
        
        self.sidebar_layout.addSpacing(10)
        
        self.color_space_label = QLabel(Strings.SELECT_COLOR_SPACE)
        self.color_space_label.setStyleSheet(f"color: {Colors.TEXT.name()}; font-weight: bold; border: none;")
        self.sidebar_layout.addWidget(self.color_space_label)
        
        self.color_space_combo = QComboBox()
        for cs_name in ColorSpaces.AVAILABLE.keys():
            self.color_space_combo.addItem(cs_name, cs_name)
        self.style_combo(self.color_space_combo)
        self.sidebar_layout.addWidget(self.color_space_combo)
        
        self.sidebar_layout.addSpacing(10)
        
        self.trace_checkbox = QCheckBox(Strings.ENABLE_TRACE)
        self.trace_checkbox.stateChanged.connect(lambda: self.trace_options.setVisible(self.trace_checkbox.isChecked()))
        self.trace_checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {Colors.TEXT.name()};
                border: none;
                spacing: 10px;
                font-size: 13px;
            }}
            QCheckBox::indicator {{
                width: 20px;
                height: 20px;
                border: 2px solid {Colors.PRIMARY.name()};
                border-radius: 4px;
                background-color: {Colors.BACKGROUND.name()};
            }}
            QCheckBox::indicator:checked {{
                background-color: {Colors.PRIMARY.name()};
                image: url(none);
                border: 2px solid {Colors.PRIMARY.name()};
            }}
            QCheckBox::indicator:checked:after {{
                content: "✓";
                color: white;
                font-size: 16px;
                font-weight: bold;
            }}
        """)
        self.sidebar_layout.addWidget(self.trace_checkbox)
        
        self.trace_options = QWidget()
        trace_options_layout = QVBoxLayout(self.trace_options)
        trace_options_layout.setContentsMargins(20, 0, 0, 0)
        trace_options_layout.setSpacing(8)
        
        color_label = QLabel(Strings.TRACE_COLOR)
        color_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px; border: none;")
        trace_options_layout.addWidget(color_label)
        
        self.trace_color_combo = QComboBox()
        self.trace_color_combo.addItem("Purple", (138, 43, 226))
        self.trace_color_combo.addItem("Red", (255, 0, 0))
        self.trace_color_combo.addItem("Green", (0, 255, 0))
        self.trace_color_combo.addItem("Blue", (0, 0, 255))
        self.trace_color_combo.addItem("Yellow", (255, 255, 0))
        self.trace_color_combo.addItem("Cyan", (0, 255, 255))
        self.trace_color_combo.addItem("White", (255, 255, 255))
        self.style_combo(self.trace_color_combo)
        trace_options_layout.addWidget(self.trace_color_combo)
        
        thickness_label = QLabel(Strings.TRACE_THICKNESS)
        thickness_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px; border: none;")
        trace_options_layout.addWidget(thickness_label)
        
        self.trace_thickness_combo = QComboBox()
        self.trace_thickness_combo.addItem("Thin (1px)", 1)
        self.trace_thickness_combo.addItem("Normal (2px)", 2)
        self.trace_thickness_combo.addItem("Thick (3px)", 3)
        self.trace_thickness_combo.addItem("Very Thick (5px)", 5)
        self.trace_thickness_combo.setCurrentIndex(1)
        self.style_combo(self.trace_thickness_combo)
        trace_options_layout.addWidget(self.trace_thickness_combo)
        
        self.trace_options.setVisible(False)
        self.sidebar_layout.addWidget(self.trace_options)
        
        self.sidebar_layout.addSpacing(20)
        
        self.start_btn = self.create_button(Strings.START_TRACKING, self.start_tracking)
        self.start_btn.setEnabled(False)
        self.sidebar_layout.addWidget(self.start_btn)
        
        self.benchmark_btn = self.create_button(Strings.BENCHMARK_MODE, self.start_benchmark_mode)
        self.benchmark_btn.setEnabled(False)
        self.benchmark_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.WARNING.name()};
                color: {Colors.BACKGROUND.name()};
                border: none;
                border-radius: 6px;
                padding: 12px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT.name()};
            }}
            QPushButton:pressed {{
                background-color: {Colors.PRIMARY.name()};
            }}
            QPushButton:disabled {{
                background-color: {Colors.SURFACE.name()};
                color: {Colors.TEXT_SECONDARY.name()};
            }}
        """)
        self.sidebar_layout.addWidget(self.benchmark_btn)
        
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {Colors.SUCCESS.name()}; font-size: 11px; border: none;")
        self.status_label.setWordWrap(True)
        self.sidebar_layout.addWidget(self.status_label)
        
        self.sidebar_layout.addStretch()
        
        self.on_model_changed(self.model_combo.currentText())
        
        return sidebar
    
    def create_button(self, text, callback):
        btn = QPushButton(text)
        btn.clicked.connect(callback)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.PRIMARY.name()};
                color: {Colors.TEXT.name()};
                border: none;
                border-radius: 6px;
                padding: 12px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT.name()};
            }}
            QPushButton:pressed {{
                background-color: {Colors.SECONDARY.name()};
            }}
            QPushButton:disabled {{
                background-color: {Colors.SURFACE.name()};
                color: {Colors.TEXT_SECONDARY.name()};
            }}
        """)
        return btn
    
    def style_combo(self, combo):
        combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {Colors.BACKGROUND.name()};
                color: {Colors.TEXT.name()};
                border: 2px solid {Colors.PRIMARY.name()};
                border-radius: 6px;
                padding: 8px;
                font-size: 12px;
            }}
            QComboBox:hover {{
                border-color: {Colors.ACCENT.name()};
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 7px solid {Colors.PRIMARY.name()};
                margin-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {Colors.SURFACE.name()};
                color: {Colors.TEXT.name()};
                selection-background-color: {Colors.PRIMARY.name()};
                border: 1px solid {Colors.PRIMARY.name()};
            }}
        """)
    
    def apply_dark_theme(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {Colors.BACKGROUND.name()};
            }}
            QLabel {{
                color: {Colors.TEXT.name()};
            }}
        """)
    
    def on_model_changed(self, model_name):
        all_models = {**Models.OPENCV_TRACKERS, **Models.DEEP_TRACKERS}
        if model_name in all_models:
            info = all_models[model_name]
            self.model_desc.setText(info["description"])
            
            supports_color_space = info["supports_color_space"]
            self.color_space_label.setVisible(supports_color_space)
            self.color_space_combo.setVisible(supports_color_space)
    
    def select_video(self):
        video_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "videa")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            Strings.SELECT_VIDEO,
            video_dir,
            "Video Files (*.mp4 *.avi *.mov)"
        )
        
        if file_path:
            self.video_path = file_path
            filename = os.path.basename(file_path)
            self.video_info.setText(f"Video: {filename}")
            self.start_btn.setEnabled(True)
            self.benchmark_btn.setEnabled(True)
            
            cap = cv.VideoCapture(file_path)
            ret, frame = cap.read()
            if ret:
                frame = cv.resize(frame, (Sizes.SIDEBAR_WIDTH - 40, Sizes.THUMBNAIL_HEIGHT - 4))
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.thumbnail_label.setPixmap(pixmap)
                self.thumbnail_label.show()
            cap.release()
    
    def start_tracking(self):
        print("[GUI] start_tracking() called")
        if not self.video_path:
            QMessageBox.warning(self, Strings.ERROR, Strings.NO_VIDEO)
            return
        
        print(f"[GUI] Video path: {self.video_path}")
        self.selected_model = self.model_combo.currentData()
        self.selected_color_space = self.color_space_combo.currentData()
        self.enable_trace = self.trace_checkbox.isChecked()
        
        print(f"[GUI] Selected model: {self.selected_model}")
        
        try:
            self.tracker = self.create_tracker()
            print(f"[GUI] Tracker created: {self.tracker}")
            if self.tracker is None:
                QMessageBox.critical(self, Strings.ERROR, f"Failed to initialize tracker: {self.selected_model}")
                return
        except Exception as e:
            QMessageBox.critical(self, Strings.ERROR, f"Tracker error: {str(e)}")
            import traceback
            traceback.print_exc()
            return
        
        self.video_processor = VideoProcessor(self.video_path, self.tracker, self.enable_trace)
        if not self.video_processor.open():
            QMessageBox.critical(self, Strings.ERROR, "Cannot open video file")
            return
        
        first_frame = self.video_processor.read_frame()
        if first_frame is None:
            QMessageBox.critical(self, Strings.ERROR, "Cannot read video")
            return
        
        self.temp_frame = first_frame.copy()
        self.roi_mode = True
        self.roi_start = None
        self.roi_end = None
        
        self.status_label.setText("Draw rectangle on video to select object")
        self.display_frame(first_frame)
        self.video_label.installEventFilter(self)
        self.start_btn.setVisible(False)
        self.video_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.color_space_combo.setEnabled(False)
        self.trace_checkbox.setEnabled(False)
    
    def create_tracker(self):
        model_name = self.selected_model
        color_space = self.selected_color_space
        
        if model_name == "CamShift":
            return CamShiftTracker(color_space)
        elif model_name == "MeanShift":
            return MeanShiftTracker(color_space)
        elif model_name == "CSRT":
            return CSRTTracker(color_space)
        elif model_name == "KCF":
            return KCFTracker(color_space)
        elif model_name == "MOSSE":
            return MOSSETracker()
        elif model_name == "MIL":
            return MILTracker()
        elif model_name == "YOLOv8":
            return YOLOTracker()
        return None
    
    def update_frame(self):
        if self.is_paused:
            return
            
        frame = self.video_processor.read_frame()
        if frame is None:
            self.stop_tracking()
            return
        
        success, result = self.tracker.update(frame)
        
        if success:
            frame = self.video_processor.draw_bbox(frame, result)
            
            if self.trace_drawer:
                center = self.tracker.get_center(self.tracker.bbox if hasattr(self.tracker, 'bbox') else result)
                self.trace_drawer.add_point(center)
                frame = self.trace_drawer.draw(frame)
        
        self.display_frame(frame)
    
    def toggle_pause(self):
        if self.is_paused:
            self.is_paused = False
            self.pause_resume_btn.setText(Strings.STOP_TRACKING)
            self.status_label.setText(Strings.TRACKING_ACTIVE)
        else:
            self.is_paused = True
            self.pause_resume_btn.setText(Strings.RESUME_TRACKING)
            self.status_label.setText("Tracking paused")
    
    def display_frame(self, frame):
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
    
    def stop_tracking(self):
        self.timer.stop()
        self.is_tracking = False
        self.is_paused = False
        if self.video_processor:
            self.video_processor.release()
        
        self.status_label.setText("Tracking complete - click Start or Reset")
        self.pause_resume_btn.setVisible(False)
        self.cancel_btn.setVisible(False)
        self.reset_btn.setVisible(True)
        self.start_btn.setVisible(True)
        self.start_btn.setEnabled(True)
        self.video_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.color_space_combo.setEnabled(True)
        self.trace_checkbox.setEnabled(True)
        self.trace_color_combo.setEnabled(True)
        self.trace_thickness_combo.setEnabled(True)
    
    def cancel_tracking(self):
        self.timer.stop()
        self.is_tracking = False
        self.is_paused = False
        if self.video_processor:
            self.video_processor.release()
            self.video_processor = None
        
        if self.tracker and hasattr(self.tracker, 'reset'):
            self.tracker.reset()
        self.tracker = None
        self.trace_drawer = None
        self.roi_mode = False
        self.roi_start = None
        self.roi_end = None
        self.temp_frame = None
        
        self.status_label.setText("Tracking cancelled")
        self.pause_resume_btn.setVisible(False)
        self.cancel_btn.setVisible(False)
        self.reset_btn.setVisible(False)
        self.video_label.setText(Strings.SELECT_VIDEO_FIRST)
        self.start_btn.setVisible(True)
        self.start_btn.setEnabled(True if self.video_path else False)
        self.benchmark_btn.setVisible(True)
        self.benchmark_btn.setEnabled(True if self.video_path else False)
        self.video_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.color_space_combo.setEnabled(True)
        self.trace_checkbox.setEnabled(True)
        self.trace_color_combo.setEnabled(True)
        self.trace_thickness_combo.setEnabled(True)
    
    def reset_tracking(self):
        self.timer.stop()
        self.pause_resume_btn.setVisible(False)
        self.cancel_btn.setVisible(False)
        self.reset_btn.setVisible(False)
        self.video_label.setText(Strings.SELECT_VIDEO_FIRST)
        self.status_label.setText("")
        self.start_btn.setVisible(True)
        self.start_btn.setEnabled(True if self.video_path else False)
        self.benchmark_btn.setVisible(True)
        self.benchmark_btn.setEnabled(True if self.video_path else False)
        self.video_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.color_space_combo.setEnabled(True)
        self.trace_checkbox.setEnabled(True)
        self.trace_color_combo.setEnabled(True)
        self.trace_thickness_combo.setEnabled(True)
        
        if self.video_processor:
            self.video_processor.release()
            self.video_processor = None
        
        if self.tracker and hasattr(self.tracker, 'reset'):
            self.tracker.reset()
        self.tracker = None
        self.trace_drawer = None
        self.is_tracking = False
        self.is_paused = False
        self.roi_mode = False
        self.roi_start = None
        self.roi_end = None
        self.temp_frame = None
    
    def eventFilter(self, obj, event):
        if obj == self.video_label and self.roi_mode:
            if event.type() == event.Type.MouseButtonPress:
                pos = self.map_to_frame_coords(event.pos())
                if pos:
                    self.roi_start = pos
                    self.roi_end = pos
                return True
            elif event.type() == event.Type.MouseMove and self.roi_start:
                pos = self.map_to_frame_coords(event.pos())
                if pos:
                    self.roi_end = pos
                    self.draw_roi_rectangle()
                return True
            elif event.type() == event.Type.MouseButtonRelease and self.roi_start:
                pos = self.map_to_frame_coords(event.pos())
                if pos:
                    self.roi_end = pos
                    self.finalize_roi()
                return True
        return super().eventFilter(obj, event)
    
    def map_to_frame_coords(self, label_pos):
        if self.temp_frame is None:
            return None
        
        pixmap = self.video_label.pixmap()
        if not pixmap:
            return None
        
        label_rect = self.video_label.rect()
        pixmap_rect = pixmap.rect()
        
        x_offset = (label_rect.width() - pixmap_rect.width()) // 2
        y_offset = (label_rect.height() - pixmap_rect.height()) // 2
        
        x = label_pos.x() - x_offset
        y = label_pos.y() - y_offset
        
        if x < 0 or y < 0 or x >= pixmap_rect.width() or y >= pixmap_rect.height():
            return None
        
        frame_h, frame_w = self.temp_frame.shape[:2]
        scale_x = frame_w / pixmap_rect.width()
        scale_y = frame_h / pixmap_rect.height()
        
        return QPoint(int(x * scale_x), int(y * scale_y))
    
    def draw_roi_rectangle(self):
        if not self.roi_start or not self.roi_end or self.temp_frame is None:
            return
        
        display_frame = self.temp_frame.copy()
        x1, y1 = self.roi_start.x(), self.roi_start.y()
        x2, y2 = self.roi_end.x(), self.roi_end.y()
        
        cv.rectangle(display_frame, (x1, y1), (x2, y2), Colors.TRACKER_BOX, 2)
        self.display_frame(display_frame)
    
    def finalize_roi(self):
        if not self.roi_start or not self.roi_end:
            return
        
        x1, y1 = self.roi_start.x(), self.roi_start.y()
        x2, y2 = self.roi_end.x(), self.roi_end.y()
        
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        
        if w < 5 or h < 5:
            QMessageBox.warning(self, Strings.ERROR, "Region too small")
            if self.benchmark_mode:
                self.cancel_benchmark()
            else:
                self.reset_tracking()
            return
        
        frame_h, frame_w = self.temp_frame.shape[:2]
        if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
            QMessageBox.warning(self, Strings.ERROR, "Region outside frame bounds")
            if self.benchmark_mode:
                self.cancel_benchmark()
            else:
                self.reset_tracking()
            return
        
        bbox = (x, y, w, h)
        
        if self.benchmark_mode:
            self.benchmark_roi = bbox
            self.run_benchmark()
            return
        
        self.roi_mode = False
        self.video_label.removeEventFilter(self)
        
        self.video_processor.reset()
        first_frame = self.video_processor.read_frame()
        
        try:
            print(f"[GUI] Initializing tracker with bbox: {bbox}")
            self.tracker.init(first_frame, bbox)
            print(f"[GUI] Tracker initialized successfully")
        except Exception as e:
            QMessageBox.critical(self, Strings.ERROR, f"Tracker initialization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            self.reset_tracking()
            return
        
        trace_color = self.trace_color_combo.currentData()
        trace_thickness = self.trace_thickness_combo.currentData()
        self.trace_drawer = TraceDrawer(trace_color, trace_thickness) if self.enable_trace else None
        
        self.status_label.setText(Strings.TRACKING_ACTIVE)
        self.is_tracking = True
        self.is_paused = False
        self.pause_resume_btn.setText(Strings.STOP_TRACKING)
        self.pause_resume_btn.setVisible(True)
        self.cancel_btn.setVisible(True)
        self.timer.start(int(1000 / 30))
    
    def closeEvent(self, event):
        self.timer.stop()
        if self.video_processor:
            self.video_processor.release()
        cv.destroyAllWindows()
        event.accept()
    
    def show_normal_mode(self):
        self.benchmark_mode = False
        
        if hasattr(self, 'back_btn') and self.back_btn:
            self.sidebar_layout.removeWidget(self.back_btn)
            self.back_btn.deleteLater()
            self.back_btn = None
        
        self.video_area.setVisible(True)
        self.video_label.setVisible(True)
        self.video_label.setText(Strings.SELECT_VIDEO_FIRST if not self.video_path else "")
        self.video_label.setStyleSheet(f"background-color: {Colors.BACKGROUND.name()}; border: none; font-size: 18px; color: {Colors.TEXT_SECONDARY.name()};")
        
        self.video_btn.setVisible(True)
        self.video_btn.setEnabled(True)
        self.thumbnail_label.setVisible(True if self.video_path else False)
        self.video_info.setVisible(True)
        
        self.model_label.setVisible(True)
        self.model_combo.setVisible(True)
        self.model_combo.setEnabled(True)
        self.model_desc.setVisible(True)
        
        current_model = self.model_combo.currentText()
        supports_color_space = Models.OPENCV_TRACKERS.get(current_model, {}).get('supports_color_space', False) or \
                              Models.DEEP_TRACKERS.get(current_model, {}).get('supports_color_space', False)
        self.color_space_label.setVisible(supports_color_space)
        self.color_space_combo.setVisible(supports_color_space)
        self.color_space_combo.setEnabled(True)
        
        self.trace_checkbox.setVisible(True)
        self.trace_checkbox.setEnabled(True)
        self.trace_options.setVisible(self.trace_checkbox.isChecked())
        self.trace_color_combo.setEnabled(True)
        self.trace_thickness_combo.setEnabled(True)
        
        self.start_btn.setVisible(True)
        self.start_btn.setEnabled(True if self.video_path else False)
        self.benchmark_btn.setVisible(True)
        self.benchmark_btn.setEnabled(True if self.video_path else False)
        
        self.status_label.setText("")
        self.status_label.setStyleSheet(f"color: {Colors.SUCCESS.name()}; font-size: 11px; border: none;")
    
    def show_benchmark_mode(self):
        self.benchmark_mode = True
        
        self.back_btn = self.create_button("← Back to Menu", self.cancel_benchmark)
        self.back_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.SURFACE.name()};
                color: {Colors.TEXT.name()};
                border: 2px solid {Colors.PRIMARY.name()};
                border-radius: 8px;
                padding: 12px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {Colors.PRIMARY.name()};
                color: white;
            }}
        """)
        self.sidebar_layout.insertWidget(1, self.back_btn)
        
        self.video_area.setVisible(True)
        self.video_label.setVisible(True)
        self.video_label.setText("")
        self.video_label.setStyleSheet("")
        
        self.video_btn.setVisible(False)
        self.model_label.setVisible(False)
        self.model_combo.setVisible(False)
        self.model_desc.setVisible(False)
        self.color_space_label.setVisible(False)
        self.color_space_combo.setVisible(False)
        self.trace_checkbox.setVisible(False)
        self.trace_options.setVisible(False)
        self.start_btn.setVisible(False)
        self.benchmark_btn.setVisible(False)
        
        self.status_label.setText("Select ROI to start crazy benchmark you really want to see!")
        self.status_label.setVisible(True)
        self.status_label.setStyleSheet(f"color: {Colors.WARNING.name()}; font-size: 14px; font-weight: bold; border: none;")
    
    def start_benchmark_mode(self):
        if not self.video_path:
            QMessageBox.warning(self, Strings.ERROR, Strings.NO_VIDEO)
            return
        
        self.show_benchmark_mode()
        
        capture = cv.VideoCapture(self.video_path)
        ret, first_frame = capture.read()
        capture.release()
        
        if not ret:
            QMessageBox.critical(self, Strings.ERROR, "Cannot read video")
            self.cancel_benchmark()
            return
        
        self.temp_frame = first_frame.copy()
        self.roi_mode = True
        self.roi_start = None
        self.roi_end = None
        
        self.status_label.setText("Draw rectangle to select tracking region")
        self.display_frame(first_frame)
        self.video_label.installEventFilter(self)
    
    def run_benchmark(self):
        self.video_label.removeEventFilter(self)
        self.roi_mode = False
        
        progress_widget = QFrame()
        progress_widget.setStyleSheet(f"background-color: {Colors.BACKGROUND.name()};")
        progress_layout = QVBoxLayout(progress_widget)
        progress_layout.setContentsMargins(40, 40, 40, 40)
        progress_layout.setSpacing(20)
        
        logo_label = QLabel()
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "appIcon.png")
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path)
            scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(logo_label)
        
        title_label = QLabel("Benchmark in Progress")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setStyleSheet(f"color: {Colors.PRIMARY.name()};")
        title_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(title_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid {Colors.PRIMARY.name()};
                border-radius: 5px;
                text-align: center;
                color: {Colors.TEXT.name()};
                background-color: {Colors.SURFACE.name()};
            }}
            QProgressBar::chunk {{
                background-color: {Colors.PRIMARY.name()};
            }}
        """)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Initializing...")
        self.progress_label.setStyleSheet(f"color: {Colors.TEXT.name()}; font-size: 14px;")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setWordWrap(True)
        progress_layout.addWidget(self.progress_label)
        
        progress_layout.addSpacing(20)
        
        self.cancel_benchmark_btn = QPushButton("Cancel Benchmark")
        self.cancel_benchmark_btn.clicked.connect(self.cancel_benchmark)
        self.cancel_benchmark_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ERROR.name()};
                color: {Colors.TEXT.name()};
                padding: 12px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {Colors.WARNING.name()};
            }}
        """)
        progress_layout.addWidget(self.cancel_benchmark_btn)
        
        progress_layout.addStretch()
        
        self.video_label.clear()
        self.video_label.setText("")
        self.video_label.setStyleSheet(f"background-color: {Colors.BACKGROUND.name()}; border: none;")
        central_widget = self.centralWidget()
        main_layout = central_widget.layout()
        
        for i in reversed(range(main_layout.count())):
            widget = main_layout.itemAt(i).widget()
            if widget and widget not in (self.video_label, self.sidebar, self.video_area):
                widget.setVisible(False)
        
        main_layout.addWidget(progress_widget)
        
        all_models = self.create_all_trackers()
        
        self.benchmark_engine = BenchmarkEngine(self.video_path, self.benchmark_roi, all_models)
        
        def progress_callback(current, total, message):
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
            self.progress_label.setText(f"{message}\nProgress: {current}/{total} configurations")
            QApplication.processEvents()
        
        try:
            results = self.benchmark_engine.run_benchmark(progress_callback)
            self.show_benchmark_results()
        except Exception as e:
            QMessageBox.critical(self, Strings.ERROR, f"Benchmark failed: {str(e)}")
            import traceback
            traceback.print_exc()
            self.cancel_benchmark()
    
    def create_all_trackers(self):
        trackers = []
        
        for color_space in ["HSV", "RGB", "YCbCr", "LAB"]:
            trackers.append((f"CamShift ({color_space})", CamShiftTracker(color_space)))
        
        for color_space in ["HSV", "RGB", "YCbCr", "LAB"]:
            trackers.append((f"MeanShift ({color_space})", MeanShiftTracker(color_space)))
        
        trackers.append(("CSRT", CSRTTracker()))
        trackers.append(("KCF", KCFTracker()))
        trackers.append(("MOSSE", MOSSETracker()))
        trackers.append(("MIL", MILTracker()))
        trackers.append(("YOLOv8", YOLOTracker()))
        
        return trackers
    
    def show_benchmark_results(self):
        results_widget = QFrame()
        results_widget.setStyleSheet(f"background-color: {Colors.BACKGROUND.name()};")
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(40, 40, 40, 40)
        results_layout.setSpacing(20)
        
        title = QLabel(Strings.BENCHMARK_COMPLETE)
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet(f"color: {Colors.SUCCESS.name()};")
        title.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(title)
        
        summary = self.benchmark_engine.get_error_summary()
        total_configs = len(summary)
        best_model = summary[0] if summary else None
        avg_pos = sum(item['mse_position'] for item in summary) / total_configs if total_configs else 0
        avg_area = sum(item['mse_area'] for item in summary) / total_configs if total_configs else 0
        
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(16)
        
        def make_card(title_text, value_text, color):
            card = QFrame()
            card.setStyleSheet(f"""
                QFrame {{
                    background-color: {Colors.SURFACE.name()};
                    border: 2px solid {Colors.PRIMARY.name()};
                    border-radius: 10px;
                    padding: 12px;
                }}
            """)
            card_layout = QVBoxLayout(card)
            card_layout.setSpacing(6)
            
            title_label = QLabel(title_text)
            title_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px; border: none;")
            value_label = QLabel(value_text)
            value_label.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold; border: none;")
            
            card_layout.addWidget(title_label)
            card_layout.addWidget(value_label)
            return card
        
        best_name = best_model['model'] if best_model else "-"
        cards_layout.addWidget(make_card("Best model", best_name, Colors.SUCCESS.name()))
        cards_layout.addWidget(make_card("Avg position RMSE", f"{avg_pos:.2f} px", Colors.ACCENT.name()))
        cards_layout.addWidget(make_card("Avg area RMSE", f"{avg_area:.2f} px²", Colors.WARNING.name()))
        results_layout.addLayout(cards_layout)
        
        table = QTableWidget(total_configs, 4)
        table.setHorizontalHeaderLabels(["Rank", "Model", "Position RMSE (px)", "Area RMSE (px²)"])
        table.setSortingEnabled(False)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(True)
        table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {Colors.SURFACE.name()};
                color: {Colors.TEXT.name()};
                border: 2px solid {Colors.PRIMARY.name()};
                border-radius: 8px;
                gridline-color: {Colors.PRIMARY.name()};
                font-size: 12px;
            }}
            QTableWidget::item {{
                color: {Colors.TEXT.name()};
                background-color: {Colors.SURFACE.name()};
            }}
            QTableWidget::item:alternate {{
                background-color: {Colors.BACKGROUND.name()};
            }}
            QHeaderView::section {{
                background-color: {Colors.BACKGROUND.name()};
                color: {Colors.TEXT.name()};
                border: none;
                padding: 6px;
                font-weight: bold;
            }}
            QTableWidget::item:selected {{
                background-color: {Colors.PRIMARY.name()};
                color: white;
            }}
        """)
        
        medal_colors = [QColor(255, 215, 0), QColor(192, 192, 192), QColor(205, 127, 50)]
        default_color = Colors.TEXT.name()
        for i, model_info in enumerate(summary):
            rank_item = QTableWidgetItem(str(i + 1))
            model_item = QTableWidgetItem(model_info['model'])
            pos_item = QTableWidgetItem(f"{model_info['mse_position']:.2f}")
            area_item = QTableWidgetItem(f"{model_info['mse_area']:.2f}")
            
            for item in (rank_item, model_item, pos_item, area_item):
                item.setTextAlignment(Qt.AlignCenter)
                item.setForeground(QColor(default_color))
                if i < 3:
                    item.setForeground(medal_colors[i])
            
            table.setItem(i, 0, rank_item)
            table.setItem(i, 1, model_item)
            table.setItem(i, 2, pos_item)
            table.setItem(i, 3, area_item)

        table.setSortingEnabled(True)
        
        results_layout.addWidget(table, 1)
        
        buttons_layout = QHBoxLayout()
        
        export_btn = self.create_button(Strings.EXPORT_RESULTS, self.export_benchmark_results)
        buttons_layout.addWidget(export_btn)
        
        done_btn = self.create_button("Done", self.cancel_benchmark)
        buttons_layout.addWidget(done_btn)
        
        results_layout.addLayout(buttons_layout)
        
        self.video_area.setVisible(False)
        self.video_label.clear()
        self.video_label.setStyleSheet(f"background-color: {Colors.BACKGROUND.name()}; border: none;")
        central_widget = self.centralWidget()
        main_layout = central_widget.layout()
        
        for i in reversed(range(main_layout.count())):
            widget = main_layout.itemAt(i).widget()
            if widget and widget not in (self.video_label, self.sidebar, self.video_area):
                widget.setVisible(False)
                main_layout.removeWidget(widget)
                widget.deleteLater()
        
        main_layout.addWidget(results_widget)
    
    def export_benchmark_results(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Benchmark Results",
            "benchmark_results.json",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                output_path = self.benchmark_engine.export_to_json(file_path)
                QMessageBox.information(self, "Success", f"Results exported to:\n{output_path}")
            except Exception as e:
                QMessageBox.critical(self, Strings.ERROR, f"Failed to export: {str(e)}")
    
    def cancel_benchmark(self):
        self.benchmark_roi = None
        self.roi_mode = False
        self.roi_start = None
        self.roi_end = None
        self.temp_frame = None
        
        central_widget = self.centralWidget()
        main_layout = central_widget.layout()
        
        for i in reversed(range(main_layout.count())):
            widget = main_layout.itemAt(i).widget()
            if widget and widget not in (self.video_label, self.sidebar, self.video_area):
                widget.setVisible(False)
                main_layout.removeWidget(widget)
                widget.deleteLater()
        
        self.show_normal_mode()

def run_application():
    app = QApplication(sys.argv)
    
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "appIcon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_application()
