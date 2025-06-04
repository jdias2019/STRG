import json
import socket

from typing import Optional, Sequence, Literal

import cv2

import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

# Constantes para configuração e valores padrão
HAND_CONFIDENCE_THRESHOLD: float = 0.5 # limiar de confiança para considerar uma mão como detetada (esquerda/direita)
MP_MODEL_COMPLEXITY: int = 0 # complexidade do modelo MediaPipe Hands (0 ou 1)
MP_MAX_NUM_HANDS: int = 2 # número máximo de mãos a detetar
MP_MIN_DETECTION_CONFIDENCE: float = 0.5 # confiança mínima para deteção inicial de mãos
MP_MIN_TRACKING_CONFIDENCE: float = 0.5 # confiança mínima para rastreamento de mãos
DEFAULT_SERVER_IP: str = "127.0.0.1" # endereço IP padrão do servidor UDP
DEFAULT_SERVER_PORT: int = 4242 # porta padrão do servidor UDP


def extract_hand_type_index(
    multi_handedness: Sequence, # lista de classificações de mãos (esquerda/direita) fornecida pelo MediaPipe
    hand_type: Literal["left", "right"], # tipo de mão a procurar ("left" ou "right")
) -> int:

    # extrai a label (e.g., 'Left', 'Right') e a pontuação de cada classificação de mão
    hand_classification = [hand.classification[0] for hand in multi_handedness]

    # filtra as mãos para encontrar aquelas que correspondem ao hand_type e têm pontuação > threshold
    hands = filter(
        lambda x: x.label.lower() == hand_type and x.score > HAND_CONFIDENCE_THRESHOLD,
        hand_classification
    )
    # ordena as mãos encontradas pela pontuação, da mais alta para a mais baixa
    hands = sorted(hands, key=lambda x: x.score, reverse=True)

    if len(hands) == 0:
        return -1 # nenhuma mão do tipo especificado encontrada com confiança suficiente

    # retorna o índice da primeira (e melhor) mão encontrada na lista original de classificações
    return hand_classification.index(hands[0])


def extract_left_right_hand_coords(
    multi_hand_landmarks: Optional[Sequence], # lista de landmarks para cada mão detetada
    multi_handedness: Sequence, # lista de classificações de mãos (esquerda/direita)
) -> dict:

    hand_coords = {
        "left": None,
        "right": None,
    }

    if multi_hand_landmarks is None:
        return hand_coords # retorna dicionário com None se não houver landmarks
    
    for hand_type in ["left", "right"]:
        # obtém o índice da mão específica (esquerda ou direita) na lista de mãos detetadas
        hand_idx = extract_hand_type_index(multi_handedness, hand_type)

        if hand_idx == -1:
            continue # passa para o próximo tipo de mão se esta não for encontrada

        # obtém os landmarks para a mão identificada
        hand_lms = multi_hand_landmarks[hand_idx]

        # extrai as coordenadas [x, y, z] de cada landmark da mão
        hand_coords[hand_type] = [
            [landmark.x, landmark.y, landmark.z] for landmark in hand_lms.landmark
        ]

    return hand_coords


def run_hand_tracking_server(
    server_ip: str, # endereço IP do servidor que receberá os dados via UDP
    server_port: int, # porta do servidor UDP
) -> None:
    # configura o socket UDP para enviar os dados das coordenadas das mãos
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # abre o feed de vídeo da webcam (tenta índice 0, depois 1)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Could not open webcam (tried indices 0 and 1).") # erro em inglês, ok
            return
    print("Info: Webcam opened successfully.") # informação em inglês

    # cria o modelo de rastreamento de mãos do MediaPipe
    with mp_hands.Hands(
        model_complexity=MP_MODEL_COMPLEXITY,
        max_num_hands=MP_MAX_NUM_HANDS,
        min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
    ) as hands:
        print(f"Info: Hand tracking server started. Sending data to {server_ip}:{server_port}. Press 'q' to quit.")
        while cap.isOpened():
            # obtém um frame da webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: failed to capture image from webcam") # erro em inglês, ok
                break

            # o MediaPipe Hands requer imagens RGB, enquanto o OpenCV captura em BGR
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # processa o frame para detetar mãos
            results = hands.process(frame_rgb)

            # extrai as coordenadas das mãos em um dicionário:
            # hand_coords = {"left": [[x, y, z], ...], "right": [[x, y, z], ...]}
            if results.multi_handedness and results.multi_hand_landmarks:
                hand_coords_data = extract_left_right_hand_coords(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
                )
            else:
                # envia um dicionário vazio se nenhuma mão for detetada com handedness
                hand_coords_data = {"left": None, "right": None}

            # envia as coordenadas das mãos para o cliente via UDP
            # as coordenadas são serializadas como uma string JSON
            encoded_coords = json.dumps(hand_coords_data)
            client_socket.sendto(encoded_coords.encode(), (server_ip, server_port))

            # desenha os landmarks das mãos no frame para visualização
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, # imagem onde desenhar
                        hand_landmarks, # landmarks da mão atual
                        mp_hands.HAND_CONNECTIONS, # conexões entre os landmarks
                        # estilos de desenho padrão do MediaPipe
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # exibe o frame (espelhado horizontalmente para uma visualização mais intuitiva)
            cv2.imshow("3D Hand Tracking Viewer - Python Sender", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Info: 'q' pressed. Stopping server...")
                break

    # liberta recursos ao terminar
    cv2.destroyAllWindows() 
    if cap.isOpened():
        cap.release()
    client_socket.close()
    print("Info: Hand tracking server stopped and resources released.")


if __name__ == "__main__":
    # executa o servidor de rastreamento de mãos com o IP e porta padrão
    run_hand_tracking_server(
        server_ip=DEFAULT_SERVER_IP,
        server_port=DEFAULT_SERVER_PORT,
    )
