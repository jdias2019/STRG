import os
import sys
import shutil
import subprocess

from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QPainter, QLinearGradient, QFont
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QWidget,
)

from config import PROGRAMS_CONFIG, THEME
from ui_components import WinButton, CardFlow


def run_in_terminal(script_path: str):
    path = os.path.abspath(script_path)
    if not os.path.exists(path):
        print(f"Erro: O ficheiro não foi encontrado: {path}")
        return
    terminal = next((t for t in ["gnome-terminal", "konsole", "xterm"] if shutil.which(t)), None)
    if not terminal:
        print("Erro: Nenhum terminal compatível encontrado.")
        return
    cmd = f'cd "{os.path.dirname(path)}" && {terminal} -e "python3 \\"{os.path.basename(path)}\\""'
    subprocess.Popen(cmd, shell=True)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STRG Flow")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMinimumSize(900, 600)

        # controlo de redimensionamento e arrastar
        self._resize_margin = 15
        self._resizing = False
        self._resize_edges = Qt.Edges()
        self._drag_pos = None
        self._drag_area_height = 60

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.card_flow = CardFlow(PROGRAMS_CONFIG, central_widget)
        self.card_flow.card_selected.connect(self.on_card_selected)
        
        self.title_label = QLabel("STRG", central_widget)
        title_font = QFont("sans-serif", 48, QFont.Weight.Bold)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet(f"color: {THEME['title'].name()};")
        self.title_label.adjustSize()

        self.close_btn = WinButton("#ff5f57", "#ff7872", self.close, parent=central_widget)
        self.max_btn = WinButton("#ffbd2e", "#ffca58", self._toggle_max, parent=central_widget)
        self.min_btn = WinButton("#28c940", "#50d566", self.showMinimized, parent=central_widget)

        self._current_program_path = None
        self.on_card_selected(0)

    def on_card_selected(self, index):
        self._current_program_path = PROGRAMS_CONFIG[index]["path"]
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, THEME["bg_start"])
        gradient.setColorAt(1, THEME["bg_end"])
        painter.fillRect(self.rect(), gradient)

    def resizeEvent(self, event):
        self.card_flow.setGeometry(0, self._drag_area_height, self.width(), self.height() - self._drag_area_height)
        self.title_label.move(40, 30)
        self.close_btn.move(self.width() - 80, 20)
        self.max_btn.move(self.width() - 60, 20)
        self.min_btn.move(self.width() - 40, 20)
        super().resizeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.card_flow.set_current_index(round(self.card_flow.current_index - 1))
        elif event.key() == Qt.Key_Right:
            self.card_flow.set_current_index(round(self.card_flow.current_index + 1))
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self._current_program_path:
                run_in_terminal(self._current_program_path)
        
        for cfg in PROGRAMS_CONFIG:
            if event.key() == cfg["key"]:
                run_in_terminal(cfg["path"])
                
        super().keyPressEvent(event)

    def _toggle_max(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def closeEvent(self, event):
        event.accept()

    def _detect_edges(self, pos):
        w, h = self.width(), self.height()
        margin = self._resize_margin
        edges = Qt.Edges()
        if pos.x() < margin: edges |= Qt.LeftEdge
        if pos.x() > w - margin: edges |= Qt.RightEdge
        if pos.y() < margin: edges |= Qt.TopEdge
        if pos.y() > h - margin: edges |= Qt.BottomEdge
        return edges

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.position().toPoint()
            edges = self._detect_edges(pos)

            # priorizar arrastar sobre redimensionar pela margem superior
            is_top_edge_only = (edges == Qt.TopEdge)

            if not is_top_edge_only and edges:
                self._resizing = True
                self._resize_edges = edges
                self._drag_pos = event.globalPosition().toPoint()
                self._start_geom = self.geometry()
            elif pos.y() < self._drag_area_height:
                self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing:
            diff = event.globalPosition().toPoint() - self._drag_pos
            new_rect = QRectF(self._start_geom)
            if self._resize_edges & Qt.LeftEdge: new_rect.setLeft(self._start_geom.left() + diff.x())
            if self._resize_edges & Qt.RightEdge: new_rect.setRight(self._start_geom.right() + diff.x())
            if self._resize_edges & Qt.TopEdge: new_rect.setTop(self._start_geom.top() + diff.y())
            if self._resize_edges & Qt.BottomEdge: new_rect.setBottom(self._start_geom.bottom() + diff.y())
            self.setGeometry(new_rect.toRect())
        elif self._drag_pos and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
        else:
            edges = self._detect_edges(event.position().toPoint())
            if edges:
                if (edges == Qt.LeftEdge | Qt.TopEdge) or (edges == Qt.RightEdge | Qt.BottomEdge):
                    self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                elif (edges == Qt.RightEdge | Qt.TopEdge) or (edges == Qt.LeftEdge | Qt.BottomEdge):
                    self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                elif edges == Qt.LeftEdge or edges == Qt.RightEdge:
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
                else:
                    self.setCursor(Qt.CursorShape.SizeVerCursor)
            else:
                self.unsetCursor()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._resizing = False
        self._drag_pos = None
        self.unsetCursor()
        super().mouseReleaseEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())