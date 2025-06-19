import math
from PySide6.QtCore import (
    Qt,
    QPointF,
    Property,
    QPropertyAnimation,
    QEasingCurve,
    Signal,
    QRectF,
)
from PySide6.QtGui import QColor, QPainter, QFont, QPen
from PySide6.QtWidgets import QWidget, QLabel


class WinButton(QLabel):
    def __init__(self, color, hover_color, on_click, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)
        self._color = QColor(color)
        self._hover_color = QColor(hover_color)
        self._current_color = self._color
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._on_click = on_click

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._current_color)
        painter.drawEllipse(0, 0, self.width(), self.height())

    def enterEvent(self, event):
        self._current_color = self._hover_color
        self.update()

    def leaveEvent(self, event):
        self._current_color = self._color
        self.update()

    def mousePressEvent(self, event):
        if self._on_click:
            self._on_click()


class CardFlow(QWidget):
    card_selected = Signal(int)

    def __init__(self, programs, parent=None):
        super().__init__(parent)
        self._programs = programs
        self._cards = []
        self._current_index_float = 0.0

        for i, program in enumerate(self._programs):
            card = {
                "name": program["name"],
                "emoji": program["emoji"],
                "pos": QPointF(0, 0),
                "scale": 1.0,
                "opacity": 1.0,
            }
            self._cards.append(card)

        self._anim = QPropertyAnimation(self, b"currentIndexFloat")
        self._anim.setDuration(400)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    @Property(float)
    def currentIndexFloat(self):
        return self._current_index_float

    @currentIndexFloat.setter
    def currentIndexFloat(self, value):
        self._current_index_float = value
        self.update()

    @property
    def current_index(self):
        return round(self._current_index_float)

    def set_current_index(self, index):
        index = max(0, min(len(self._cards) - 1, index))
        if self.current_index != index:
            if self._anim.state() == QPropertyAnimation.State.Running:
                self._anim.stop()
            self._anim.setStartValue(self._current_index_float)
            self._anim.setEndValue(index)
            self._anim.start()
            self.card_selected.emit(index)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        center_x = self.width() / 2
        center_y = self.height() / 2
        card_width = 200
        card_height = 250
        spacing = 150

        for i, card in enumerate(self._cards):
            dx = i - self._current_index_float
            
            pos_x = center_x + dx * spacing - card_width / 2
            pos_y = center_y - card_height / 2
            scale = max(0, 1.0 - abs(dx) * 0.25)
            
            # evitar sobreposição incorreta de cartões
            if dx > 0:
                 pos_x += (1-scale) * card_width * 2
            
            painter.setOpacity(max(0, 1.0 - abs(dx) * 0.5))

            card_rect = QRectF(pos_x, pos_y, card_width, card_height)
            
            if abs(dx) < 0.5:
                pen = QPen(QColor(0, 150, 255, 180), 3)
                painter.setPen(pen)
            else:
                painter.setPen(Qt.PenStyle.NoPen)

            painter.setBrush(QColor(26, 34, 51, int(200 * painter.opacity())))
            painter.drawRoundedRect(card_rect, 20, 20)
            
            painter.setPen(QColor("#E0E5F0"))
            
            emoji_font = QFont("sans-serif", 80)
            painter.setFont(emoji_font)
            painter.drawText(card_rect, Qt.AlignmentFlag.AlignCenter, card["emoji"])
            
            name_font = QFont("sans-serif", 18, QFont.Weight.Bold)
            painter.setFont(name_font)
            text_rect = QRectF(card_rect.x(), card_rect.y() + card_height - 60, card_width, 50)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, card["name"])

    def mousePressEvent(self, event):
        center_x = self.width() / 2
        card_width = 200
        spacing = 150
        
        click_x = event.position().x()
        center_card_x = center_x - card_width / 2
        
        if click_x < center_card_x:
            self.set_current_index(self.current_index - 1)
        elif click_x > center_card_x + card_width:
            self.set_current_index(self.current_index + 1)
        else:
            pass 