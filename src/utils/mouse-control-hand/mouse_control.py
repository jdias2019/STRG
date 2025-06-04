import cv2
import numpy as np
# import HandTrackingModule as htm # módulo personalizado para deteção de mãos.
import time
from pynput.mouse import Button, Controller # para controlar o cursor e cliques do rato.
from typing import Tuple # para type hinting.
import tkinter as tk # Adicionado para obter as dimensões do ecrã

import sys
import os

# Adiciona o diretório 'src' ao sys.path para permitir importações absolutas de módulos dentro de 'src'.
# Obtém o caminho para o diretório 'src/utils/mouse-control-hand', sobe dois níveis para 'src'.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # src/utils/mouse-control-hand
UTILS_DIR = os.path.dirname(SCRIPT_DIR) # src/utils
SRC_DIR = os.path.dirname(UTILS_DIR) # src

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from utils import HandTrackingModule as htm


##########################
wCam, hCam = 640, 480
frameR = 85 
smoothening = 5 
#########################

pTime = 0
plocX, plocY = 0, 0 # posição anterior do cursor
clocX, clocY = 0, 0 # posição atual do cursor

# inicia o controlador do rato
mouse = Controller()

# obtém as dimensões reais do ecrã usando tkinter
try:
    root_tk = tk.Tk()
    wScr = root_tk.winfo_screenwidth()
    hScr = root_tk.winfo_screenheight()
    root_tk.destroy() # Destruir a janela root temporária
    print(f"Resolução do ecrã detectada (tkinter): {wScr}x{hScr}")
except Exception as e:
    print(f"Erro ao obter resolução do ecrã com tkinter: {e}. Usando valores padrão 1920x1080.")
    wScr, hScr = 1920, 1080


# tente usar a câmera 0 primeiro (webcam padrão)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    # se a câmera 0 não funcionar, tente a câmera 1
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Erro: Não foi possível acessar a câmera. Verifique a conexão.")
        exit()

cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)

while True:
    # 1. encontra os landmarks da mão
    success, img = cap.read()
    if not success:
        print("Erro ao capturar o frame da câmera.")
        break
        
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # print(f"lmList length: {len(lmList)}") # debug
    
    # 2. obtém as pontas dos dedos indicador e médio
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:] # ponta do dedo indicador
        x2, y2 = lmList[12][1:] # ponta do dedo médio
        
        # 3. verifica quais dedos estão levantados
        fingers = detector.fingersUp()
        # print(f"Fingers: {fingers}") # debug

        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
        (255, 0, 255), 2)
        
        # 4. apenas indicador: modo de movimentação
        if fingers and len(fingers) == 5 and fingers[1] == 1 and fingers[2] == 0: # adicionada verificação de segurança para fingers
            # print("Attempting to move mouse...") # debug
            # 5. converte coordenadas - espelhando horizontalmente
            x3 = np.interp(x1, (frameR, wCam - frameR), (wScr, 0))  # espelhado horizontalmente
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))  # não espelhado verticalmente
            
            # 6. suaviza valores
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            
            # 7. move o mouse
            try:
                # certifica que coordenadas estão dentro dos limites do ecrã
                mouse_x = min(max(int(clocX), 0), int(wScr))
                mouse_y = min(max(int(clocY), 0), int(hScr))
                # print(f"Calculated mouse position: ({mouse_x}, {mouse_y})") # debug
                
                # Move o mouse
                mouse.position = (mouse_x, mouse_y)
                
                # Destaca o ponto de controle
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY
            except Exception as e:
                print(f"Erro ao mover o mouse: {e}")
            
        # 8. indicador e médio levantados: modo de clique
        if fingers and len(fingers) == 5 and fingers[1] == 1 and fingers[2] == 1: # adicionada verificação de segurança para fingers
            # 9. encontra a distância entre os dedos
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # print(f"Distance for click: {length}") # debug
            
            # 10. clica com o mouse se a distância for pequena
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                15, (0, 255, 0), cv2.FILLED)
                try:
                    mouse.click(Button.left)
                    # pausa após clique para evitar cliques múltiplos (reduzido de 0.2 para 0.15)
                    time.sleep(0.15)
                except Exception as e:
                    print(f"Erro ao clicar com o mouse: {e}")
    
    # 11. taxa de frames
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
    (255, 0, 0), 3)
    
    # 12. exibe a imagem
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    # pressione 'q' para sair
    if key == ord('q'):
        break

# libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows() 