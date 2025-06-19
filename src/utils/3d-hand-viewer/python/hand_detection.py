import json
import socket
import time
from typing import Optional, Sequence, Literal
import numpy as np
from collections import deque

import cv2

import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

# constantes para configuração e valores padrão
HAND_CONFIDENCE_THRESHOLD: float = 0.7  # limiar de confiança para detecção
MP_MODEL_COMPLEXITY: int = 1  # complexidade do modelo mediapipe
MP_MAX_NUM_HANDS: int = 2
MP_MIN_DETECTION_CONFIDENCE: float = 0.7  # confiança mínima para detecção
MP_MIN_TRACKING_CONFIDENCE: float = 0.8  # confiança mínima para rastreamento
DEFAULT_SERVER_IP: str = "127.0.0.1"
DEFAULT_SERVER_PORT: int = 4242

# constantes de optimização de performance
TARGET_FPS: float = 60.0
SKIP_FRAMES: int = 0  # processar todos os frames
DATA_PRECISION: int = 6  # precisão dos dados de coordenadas

# constantes de suavização e estabilização
TEMPORAL_SMOOTHING_FACTOR: float = 0.3  # factor de suavização temporal
POSITION_THRESHOLD: float = 0.02  # limiar para mudanças de posição
CONFIDENCE_HISTORY_SIZE: int = 10  # tamanho do histórico de confiança
MIN_STABLE_FRAMES: int = 3  # frames mínimos para estabilidade
LANDMARK_SMOOTHING_WINDOW: int = 5  # janela de suavização de landmarks


class HandStabilizer:
    # classe para estabilizar e suavizar a detecção de mãos
    
    def __init__(self):
        self.left_hand_history = deque(maxlen=LANDMARK_SMOOTHING_WINDOW)
        self.right_hand_history = deque(maxlen=LANDMARK_SMOOTHING_WINDOW)
        self.left_confidence_history = deque(maxlen=CONFIDENCE_HISTORY_SIZE)
        self.right_confidence_history = deque(maxlen=CONFIDENCE_HISTORY_SIZE)
        self.last_stable_left = None
        self.last_stable_right = None
        self.left_stable_frames = 0
        self.right_stable_frames = 0
        
    def _smooth_landmarks(self, current_landmarks, history):
        # aplica suavização temporal aos landmarks
        if not history:
            return current_landmarks
            
        # converte landmarks para numpy array
        current_array = np.array([[lm.x, lm.y, lm.z] for lm in current_landmarks])
        
        # calcula média ponderada com o histórico
        weights = np.exp(np.linspace(-2, 0, len(history)))
        weights /= weights.sum()
        
        smoothed_array = current_array.copy()
        for i, historical_landmarks in enumerate(history):
            hist_array = np.array([[lm.x, lm.y, lm.z] for lm in historical_landmarks])
            smoothed_array += weights[i] * (hist_array - current_array) * TEMPORAL_SMOOTHING_FACTOR
            
        return smoothed_array
        
    def _is_hand_stable(self, current_landmarks, last_stable, confidence_history):
        # verifica se a mão está estável
        if last_stable is None:
            return False
            
        # verifica estabilidade da confiança
        if len(confidence_history) < MIN_STABLE_FRAMES:
            return False
            
        avg_confidence = sum(confidence_history) / len(confidence_history)
        if avg_confidence < HAND_CONFIDENCE_THRESHOLD:
            return False
            
        # verifica estabilidade da posição
        current_array = np.array([[lm.x, lm.y, lm.z] for lm in current_landmarks])
        last_array = np.array(last_stable)
        
        position_diff = np.mean(np.linalg.norm(current_array - last_array, axis=1))
        return position_diff < POSITION_THRESHOLD
        
    def stabilize_hands(self, results):
        # estabiliza e suaviza os resultados da detecção
        if not results.multi_hand_landmarks or not results.multi_handedness:
            # reduz contadores de estabilidade quando não há detecção
            self.left_stable_frames = max(0, self.left_stable_frames - 1)
            self.right_stable_frames = max(0, self.right_stable_frames - 1)
            return results
            
        stabilized_landmarks = []
        stabilized_handedness = []
        
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_type = handedness.classification[0].label.lower()
            confidence = handedness.classification[0].score
            
            if hand_type == "left":
                self.left_confidence_history.append(confidence)
                
                # aplica suavização se há histórico suficiente
                if len(self.left_hand_history) > 0:
                    smoothed_array = self._smooth_landmarks(hand_landmarks.landmark, self.left_hand_history)
                    
                    # verifica estabilidade
                    if self._is_hand_stable(hand_landmarks.landmark, self.last_stable_left, self.left_confidence_history):
                        self.left_stable_frames += 1
                    else:
                        self.left_stable_frames = 0
                        
                    # usa dados suavizados se a mão estiver estável
                    if self.left_stable_frames >= MIN_STABLE_FRAMES:
                        # cria novos landmarks com dados suavizados
                        for i, (x, y, z) in enumerate(smoothed_array):
                            hand_landmarks.landmark[i].x = float(x)
                            hand_landmarks.landmark[i].y = float(y)
                            hand_landmarks.landmark[i].z = float(z)
                        
                        self.last_stable_left = smoothed_array.tolist()
                
                self.left_hand_history.append(hand_landmarks.landmark)
                
            elif hand_type == "right":
                self.right_confidence_history.append(confidence)
                
                # Aplica suavização se há histórico suficiente
                if len(self.right_hand_history) > 0:
                    smoothed_array = self._smooth_landmarks(hand_landmarks.landmark, self.right_hand_history)
                    
                    # Verifica estabilidade
                    if self._is_hand_stable(hand_landmarks.landmark, self.last_stable_right, self.right_confidence_history):
                        self.right_stable_frames += 1
                    else:
                        self.right_stable_frames = 0
                        
                    # Só usa dados suavizados se a mão estiver estável
                    if self.right_stable_frames >= MIN_STABLE_FRAMES:
                        # Cria novos landmarks com dados suavizados
                        for i, (x, y, z) in enumerate(smoothed_array):
                            hand_landmarks.landmark[i].x = float(x)
                            hand_landmarks.landmark[i].y = float(y)
                            hand_landmarks.landmark[i].z = float(z)
                        
                        self.last_stable_right = smoothed_array.tolist()
                
                self.right_hand_history.append(hand_landmarks.landmark)
            
            stabilized_landmarks.append(hand_landmarks)
            stabilized_handedness.append(handedness)
        
        # actualiza os resultados com dados estabilizados
        results.multi_hand_landmarks = stabilized_landmarks
        results.multi_handedness = stabilized_handedness
        
        return results


def extract_hand_type_index(
    multi_handedness: Sequence,
    hand_type: Literal["left", "right"],
) -> int:
    # extrai a label e pontuação de cada classificação de mão
    hand_classification = [hand.classification[0] for hand in multi_handedness]

    # filtra as mãos que correspondem ao tipo e têm pontuação suficiente
    hands = filter(
        lambda x: x.label.lower() == hand_type and x.score > HAND_CONFIDENCE_THRESHOLD,
        hand_classification
    )
    # ordena as mãos pela pontuação
    hands = sorted(hands, key=lambda x: x.score, reverse=True)

    if len(hands) == 0:
        return -1

    # retorna o índice da melhor mão encontrada
    return hand_classification.index(hands[0])


def extract_left_right_hand_coords(
    multi_hand_landmarks: Optional[Sequence],
    multi_handedness: Sequence,
) -> dict:
    hand_coords = {
        "left": None,
        "right": None,
    }

    if multi_hand_landmarks is None:
        return hand_coords
    
    for hand_type in ["left", "right"]:
        # obtém o índice da mão específica na lista
        hand_idx = extract_hand_type_index(multi_handedness, hand_type)

        if hand_idx == -1:
            continue

        # obtém os landmarks para a mão identificada
        hand_lms = multi_hand_landmarks[hand_idx]

        # converte coordenadas para dicionário
        hand_coords[hand_type] = {
            str(i): [
                round(landmark.x, DATA_PRECISION), 
                round(landmark.y, DATA_PRECISION), 
                round(landmark.z, DATA_PRECISION)
            ]
            for i, landmark in enumerate(hand_lms.landmark)
        }

    return hand_coords


def run_hand_tracking_server(
    server_ip: str,
    server_port: int,
) -> None:
    # configura socket udp para envio de dados
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # abre feed de vídeo da webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Could not open webcam (tried indices 0 and 1).")
            return
    
    # optimiza configurações da câmera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    print("Info: Webcam opened successfully with enhanced settings.")

    # variáveis de controlo de frame rate
    target_interval = 1.0 / TARGET_FPS
    last_send_time = 0.0
    frame_count = 0
    last_hand_data = {"left": None, "right": None}
    
    # inicializa o estabilizador de mãos
    stabilizer = HandStabilizer()

    # cria modelo de rastreamento de mãos do mediapipe
    with mp_hands.Hands(
        model_complexity=MP_MODEL_COMPLEXITY,
        max_num_hands=MP_MAX_NUM_HANDS,
        min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
        static_image_mode=False,
    ) as hands:
        print(f"Info: Enhanced hand tracking server started. Sending data to {server_ip}:{server_port}")
        print(f"Info: Target FPS: {TARGET_FPS}, Enhanced stabilization enabled")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: failed to capture image from webcam")
                break

            current_time = time.time()
            frame_count += 1
            
            # processa todos os frames
            should_send = (current_time - last_send_time) >= target_interval
            
            # pré-processamento da imagem
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # aplica filtro para reduzir ruído
            frame_rgb = cv2.bilateralFilter(frame_rgb, 9, 75, 75)
            
            # processa frame para detectar mãos
            results = hands.process(frame_rgb)
            
            # aplica estabilização aos resultados
            if results.multi_handedness and results.multi_hand_landmarks:
                results = stabilizer.stabilize_hands(results)
                
                hand_coords_data = extract_left_right_hand_coords(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
                )
            else:
                hand_coords_data = {"left": None, "right": None}
            
            last_hand_data = hand_coords_data

            # envia dados na taxa controlada
            if should_send:
                encoded_coords = json.dumps(hand_coords_data, separators=(',', ':'))
                client_socket.sendto(encoded_coords.encode(), (server_ip, server_port))
                last_send_time = current_time

            # gui simplificada
            simple_frame = cv2.resize(frame, (320, 240))
            cv2.putText(simple_frame, f"FPS: {int(1.0 / (current_time - last_send_time + 0.001))}", 
                       (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(simple_frame, f"Hands: {len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0}", 
                       (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # exibe frame simplificado
            cv2.imshow("Hand Tracker", simple_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Info: 'q' pressed. Stopping server...")
                break

    # liberta recursos ao terminar
    cv2.destroyAllWindows() 
    if cap.isOpened():
        cap.release()
    client_socket.close()
    print("Info: Enhanced hand tracking server stopped and resources released.")


if __name__ == "__main__":
    run_hand_tracking_server(
        server_ip=DEFAULT_SERVER_IP,
        server_port=DEFAULT_SERVER_PORT,
    )
