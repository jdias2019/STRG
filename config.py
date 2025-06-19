from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

THEME = {
    "bg_start": QColor("#0D1018"),
    "bg_end": QColor("#1A2233"),
    "title": QColor("#E0E5F0"),
    "accent_glow": QColor(0, 150, 255, 180),
}

PROGRAMS_CONFIG = [
    {"key": Qt.Key_F1, "path": "src/main/main.py", "name": "Gestos", "emoji": "🖐️"},
    {"key": Qt.Key_F2, "path": "src/main/word_recognition/word_recognition_app.py", "name": "Palavras", "emoji": "💬"},
    {"key": Qt.Key_F3, "path": "src/utils/mouse-control-hand/mouse.py", "name": "Cursor", "emoji": "🖱️"},
    {"key": Qt.Key_F4, "path": "src/utils/volume-control-hand/volume.py", "name": "Volume", "emoji": "🔊"},
    {"key": Qt.Key_F5, "path": "src/utils/face-recon/face.py", "name": "Face ID", "emoji": "😀"},
    {"key": Qt.Key_F6, "path": "src/utils/binary-vision/binary_vision.py", "name": "Visão Binária", "emoji": "👁️"},
    {"key": Qt.Key_F7, "path": "src/utils/3d-hand-viewer/python/hand_detection.py", "name": "3D", "emoji": "🖐️🌀"},
    {"key": Qt.Key_F8, "path": "src/utils/menus/performance-menu/performance_menu.py", "name": "Performance", "emoji": "⚡"},
]
