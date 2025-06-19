import cv2
import mediapipe as mp
import time
import math
import numpy as np
from typing import List, Tuple, Union, Optional

class handDetector():
    _DEFAULT_LANDMARK_COLOR = (255, 0, 255)
    _DEFAULT_BBOX_COLOR = (0, 255, 0)
    _DEFAULT_LINE_COLOR = (255, 0, 255)
    _DEFAULT_CENTER_COLOR = (0, 0, 255)
    _DEFAULT_CIRCLE_RADIUS = 5
    _BBOX_PADDING = 20

    def __init__(self, mode: bool = False, maxHands: int = 2,
                 detectionCon: float = 0.5, trackCon: float = 0.5) -> None:
        self.mode: bool = mode
        self.maxHands: int = maxHands
        self.detectionCon: float = detectionCon
        self.trackCon: float = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds: List[int] = [4, 8, 12, 16, 20]
        self.results: Optional[mp.solutions.hands.process] = None
        self.lmList: List[List[int]] = []

    def findHands(self, img: np.ndarray, draw: bool = True) -> np.ndarray:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results and self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img: np.ndarray, handNo: int = 0, draw: bool = True) -> Tuple[List[List[int]], Optional[Tuple[int, int, int, int]]]:
        xList: List[int] = []
        yList: List[int] = []
        bbox: Optional[Tuple[int, int, int, int]] = None
        self.lmList = []
        if self.results and self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), self._DEFAULT_CIRCLE_RADIUS, self._DEFAULT_LANDMARK_COLOR, cv2.FILLED)
                if xList and yList:
                    xmin, xmax = min(xList), max(xList)
                    ymin, ymax = min(yList), max(yList)
                    bbox = (xmin, ymin, xmax, ymax)
                    if draw:
                        cv2.rectangle(img, (xmin - self._BBOX_PADDING, ymin - self._BBOX_PADDING),
                                      (xmax + self._BBOX_PADDING, ymax + self._BBOX_PADDING),
                                      self._DEFAULT_BBOX_COLOR, 2)
        return self.lmList, bbox

    def fingersUp(self) -> List[int]:
        fingers: List[int] = []
        if not self.lmList or len(self.lmList) < 21:
            return []
            
        # Thumb - special case - check if thumb tip is to the right of thumb base for right hand
        # For left hand this would be reversed, but we're assuming right hand for simplicity
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]: 
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Other fingers - check if finger tip is above finger pip (second joint)
        for id in range(1, 5):
            if len(self.lmList) > self.tipIds[id] and len(self.lmList) > self.tipIds[id] - 2:
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]: 
                    fingers.append(1)
                else:
                    fingers.append(0)
                    
        return fingers

    def findDistance(self, p1_id: int, p2_id: int, img: np.ndarray,
                     draw: bool = True, radius: int = 10, thickness: int = 2) -> Tuple[float, np.ndarray, List[int]]:
        if not self.lmList or not (0 <= p1_id < len(self.lmList) and 0 <= p2_id < len(self.lmList)):
            return 0.0, img, [0,0,0,0,0,0]
        x1, y1 = self.lmList[p1_id][1], self.lmList[p1_id][2]
        x2, y2 = self.lmList[p2_id][1], self.lmList[p2_id][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), self._DEFAULT_LINE_COLOR, thickness)
            cv2.circle(img, (x1, y1), radius, self._DEFAULT_LINE_COLOR, cv2.FILLED)
            cv2.circle(img, (x2, y2), radius, self._DEFAULT_LINE_COLOR, cv2.FILLED)
            cv2.circle(img, (cx, cy), radius, self._DEFAULT_CENTER_COLOR, cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

def main() -> None:
    previous_time: float = 0.0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            return
    detector = handDetector(detectionCon=0.8, maxHands=1)
    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList_main, bbox_main = detector.findPosition(img)
        current_time = time.time()
        if previous_time > 0:
            fps = 1 / (current_time - previous_time)
            cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 255), 3)
        previous_time = current_time
        cv2.imshow("Hand Tracking Test", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 