import cv2
import time
import numpy as np
# import HandTrackingModule as htm # módulo personalizado para deteção e rastreamento de mãos
import math
import platform # para detetar sistema operativo e aplicar controlos de áudio específicos

import sys
import os

# adiciona diretório 'src' ao sys.path para importações absolutas de módulos em 'src'
# obtém caminho para diretório 'src/utils/volume-control-hand', sobe dois níveis para 'src'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # src/utils/volume-control-hand
UTILS_DIR = os.path.dirname(SCRIPT_DIR) # src/utils
SRC_DIR = os.path.dirname(UTILS_DIR) # src

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from utils import HandTrackingModule as htm

# constantes de configuração da câmara e aplicação
WCAM_WIDTH, WCAM_HEIGHT = 640, 480 # largura e altura da janela da câmara
HAND_DETECTION_CONFIDENCE = 0.7 # confiança mínima para deteção de mãos
MAX_HANDS_TO_DETECT = 1 # número máximo de mãos a detetar

# constantes para lógica de controlo de volume baseada em gestos
MIN_HAND_DISTANCE_FOR_VOL = 50 # distância mínima entre dedos (polegar e indicador) para volume mínimo
MAX_HAND_DISTANCE_FOR_VOL = 200 # distância máxima entre dedos para volume máximo
VOL_BAR_MIN_Y = 400 # coordenada y mínima da barra de volume na ui
VOL_BAR_MAX_Y = 150 # coordenada y máxima da barra de volume na ui (invertido, y cresce para baixo)
MIN_HAND_AREA = 250 # área mínima da bounding box da mão para considerar gesto
MAX_HAND_AREA = 1000 # área máxima da bounding box da mão
VOLUME_SMOOTHING_FACTOR = 10 # fator para suavizar mudanças de volume (múltiplo de)

# constantes para controlo de áudio específico do sistema operativo
# para linux, são sugestões/exemplos e podem precisar de ajuste
LINUX_TARGET_SINK_KEYWORD = "hyperx" # palavra-chave para procurar no nome/descrição do sink de áudio no linux
LINUX_FALLBACK_SINK_INDEX = 0 # índice do sink a usar como fallback se palavra-chave não encontrada

# importações condicionais e configuração para controlo de áudio
# secção lida com diferenças entre windows e linux para controlo de volume do sistema
if platform.system() == "Windows":
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        print("Info: Windows OS detected. Using pycaw for audio control.") 
    except ImportError:
        print("Error: pycaw library not found. Please install it for Windows: pip install pycaw") 
        exit()
elif platform.system() == "Linux":
    try:
        import pulsectl
        print("Info: Linux OS detected. Using pulsectl for audio control.") 
    except ImportError:
        print("Error: pulsectl library not found. Please install it for Linux: pip install pulsectl") 
        exit()
else:
    print(f"Error: Unsupported OS: {platform.system()}. This script currently supports Windows and Linux for audio control.") 
    exit()

# inicialização da captura de vídeo
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture device. Check camera index or connection and permissions.") 
    exit()
cap.set(3, WCAM_WIDTH) # define largura do frame
cap.set(4, WCAM_HEIGHT) # define altura do frame

# inicialização do detetor de mãos
# detectioncon: confiança mínima para deteção
# maxhands: número máximo de mãos a detetar
detector = htm.handDetector(detectionCon=HAND_DETECTION_CONFIDENCE, maxHands=MAX_HANDS_TO_DETECT)

# inicialização das variáveis de controlo de áudio específicas do so
audio_controller_instance = None # instância do controlador de áudio (pycaw ou pulsectl)
# minvol e maxvol para interface do utilizador (0-100), mapeamento real feito depois
# funções set_volume_os e get_volume_os operam com escalar 0.0-1.0
os_min_volume_val = 0.0 # valor mínimo real que api do so aceita (geralmente 0.0 para escalar, ou db para pycaw)
os_max_volume_val = 1.0 # valor máximo real (geralmente 1.0 para escalar, ou db)
pulse_instance_ref = None # referência para fechar instância do pulsectl explicitamente no linux

if platform.system() == "Windows":
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        audio_controller_instance = cast(interface, POINTER(IAudioEndpointVolume))
        vol_range_db = audio_controller_instance.GetVolumeRange() # retorna (min_db, max_db, step_db)
        os_min_volume_val = vol_range_db[0] # em db
        os_max_volume_val = vol_range_db[1] # em db
        # # print(f"Debug: Windows volume range (dB): Min={os_min_volume_val}, Max={os_max_volume_val}") # para depuração

        # define volume no windows (aceita escalar 0.0 a 1.0)
        def set_volume_os(level_scalar: float) -> None:
            # setmastervolumelevelscalar lida com conversão de escalar para db internamente
            audio_controller_instance.SetMasterVolumeLevelScalar(np.clip(level_scalar, 0.0, 1.0), None)

        # obtém volume no windows (retorna escalar 0.0 a 1.0)
        def get_volume_os() -> float:
            return audio_controller_instance.GetMasterVolumeLevelScalar()
        print("Info: Windows audio controller initialized successfully.") 
    except Exception as e:
        print(f"Error: Failed to initialize pycaw audio controller on Windows: {e}") 
        exit()

elif platform.system() == "Linux":
    try:
        pulse_instance_ref = pulsectl.Pulse('volume-control-hand-app') # nome da aplicação para pulsectl
        sinks = pulse_instance_ref.sink_list()
        if not sinks:
            print("Error: No audio sinks found by pulsectl on Linux.") 
            if pulse_instance_ref: pulse_instance_ref.close()
            exit()
        
        print("Info: Available audio sinks on Linux:") 
        for i, s_info in enumerate(sinks):
            print(f"  Sink {i}: Name='{s_info.name}', Description='{s_info.description}', Index={s_info.index}") 
        
        target_sink_obj = None
        # tenta encontrar sink preferido por palavra-chave
        for s_info in sinks:
            if LINUX_TARGET_SINK_KEYWORD.lower() in s_info.name.lower() or \
               LINUX_TARGET_SINK_KEYWORD.lower() in s_info.description.lower():
                target_sink_obj = s_info
                print(f"Info: Found target sink by keyword '{LINUX_TARGET_SINK_KEYWORD}': '{target_sink_obj.name}' (Index {target_sink_obj.index})") 
                break
        
        # se não encontrar por palavra-chave, tenta por índice de fallback
        if not target_sink_obj:
            print(f"Warning: Could not find sink with keyword '{LINUX_TARGET_SINK_KEYWORD}'. Attempting to use fallback index {LINUX_FALLBACK_SINK_INDEX}.")    
            if 0 <= LINUX_FALLBACK_SINK_INDEX < len(sinks):
                target_sink_obj = sinks[LINUX_FALLBACK_SINK_INDEX]
                print(f"Info: Using sink by fallback index {LINUX_FALLBACK_SINK_INDEX}: '{target_sink_obj.name}' (Index {target_sink_obj.index})")       
            else: # se índice de fallback também inválido, usa primeiro sink disponível
                print(f"Error: Fallback sink index {LINUX_FALLBACK_SINK_INDEX} is out of range. Using the first available sink (index 0).") 
                target_sink_obj = sinks[0]
                print(f"Info: Using first available sink: '{target_sink_obj.name}' (Index {target_sink_obj.index})") 
        
        audio_controller_instance = target_sink_obj # armazena objeto sink encontrado
        os_min_volume_val = 0.0 # pulsectl usa escalar 0.0 para volume mínimo
        os_max_volume_val = 1.0 # pulsectl usa escalar 1.0 para volume máximo

        # define volume no linux (aceita escalar 0.0 a 1.0)
        def set_volume_os(level_scalar: float) -> None:
            # # print(f"Debug: Linux: Setting volume to {level_scalar:.2f} for sink '{audio_controller_instance.name}'") # para depuração
            pulse_instance_ref.volume_set_all_chans(audio_controller_instance, np.clip(level_scalar, 0.0, 1.0))
            # # vol_after_set = pulse_instance_ref.volume_get_all_chans(audio_controller_instance) # para depuração
            # # print(f"Debug: Linux: Volume for sink '{audio_controller_instance.name}' after set: {vol_after_set:.2f}") # para depuração

        # obtém volume no linux (retorna escalar 0.0 a 1.0)
        def get_volume_os() -> float:
            current_vol_scalar = pulse_instance_ref.volume_get_all_chans(audio_controller_instance)
            # # print(f"Debug: Linux: Current volume for sink '{audio_controller_instance.name}' (Index {audio_controller_instance.index}): {current_vol_scalar:.2f}") # para depuração
            return current_vol_scalar
        print(f"Info: Linux audio controller initialized for sink: '{target_sink_obj.name}'.") 
            
    except Exception as e:
        print(f"Error: Failed to initialize pulsectl or manage sinks on Linux: {e}") 
        if pulse_instance_ref:
            pulse_instance_ref.close()
        exit()

# variáveis de estado para ui e lógica de volume
previous_time = 0 # para cálculo de fps
volume_percentage_ui = 0 # percentagem de volume (0-100) para ui
volume_bar_y_ui = VOL_BAR_MIN_Y # coordenada y da barra de volume para ui
hand_area_detected = 0 # área da bounding box da mão
volume_indicator_color_ui = (255, 0, 0) # cor do indicador de volume (vermelho por defeito)

try:
    while True:
        success, img = cap.read() # lê frame da câmara
        if not success or img is None:
            print("Error: Failed to capture frame from camera. Loop terminated.") 
            break

        img = cv2.flip(img, 1) # inverte frame horizontalmente (efeito espelho)
        img = detector.findHands(img) # encontra mãos no frame
        landmark_list, bbox = detector.findPosition(img, draw=True) # obtém lista de landmarks e bounding box

        if len(landmark_list) != 0: # se mão detetada
            # filtra gesto com base na área da mão (distância da câmara)
            hand_area_detected = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100 # cálculo simplificado da área
            
            if MIN_HAND_AREA < hand_area_detected < MAX_HAND_AREA:
                # calcula distância entre polegar (landmark 4) e indicador (landmark 8)
                length, img, line_info = detector.findDistance(4, 8, img)

                # mapeia distância dos dedos para altura da barra de volume na ui e para percentagem de volume
                volume_bar_y_ui = np.interp(length, 
                                         [MIN_HAND_DISTANCE_FOR_VOL, MAX_HAND_DISTANCE_FOR_VOL], 
                                         [VOL_BAR_MIN_Y, VOL_BAR_MAX_Y])
                volume_percentage_ui = np.interp(length, 
                                               [MIN_HAND_DISTANCE_FOR_VOL, MAX_HAND_DISTANCE_FOR_VOL], 
                                               [0, 100]) # percentagem de 0 a 100

                # aplica suavização à percentagem de volume para evitar mudanças bruscas
                volume_percentage_ui = VOLUME_SMOOTHING_FACTOR * round(volume_percentage_ui / VOLUME_SMOOTHING_FACTOR)

                # verifica quais dedos estão levantados
                fingers_up = detector.fingersUp()
                
                # se dedo mindinho para baixo, define volume do sistema
                # gesto de ativação para alterar volume
                if not fingers_up[4]: # índice 4 corresponde ao dedo mindinho
                    volume_scalar_to_set_os = volume_percentage_ui / 100.0 # converte percentagem (0-100) para escalar (0.0-1.0)
                    set_volume_os(volume_scalar_to_set_os) 
                    cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED) # círculo verde no gesto
                    volume_indicator_color_ui = (0, 255, 0) # verde indica que volume está a ser definido
                else:
                    volume_indicator_color_ui = (255, 0, 0) # vermelho indica que o gesto não está ativo para mudar volume

        # desenha elementos da ui (barra de volume, percentagem)
        cv2.rectangle(img, (50, VOL_BAR_MAX_Y), (85, VOL_BAR_MIN_Y), (255, 0, 0), 3) # contorno da barra
        cv2.rectangle(img, (50, int(volume_bar_y_ui)), (85, VOL_BAR_MIN_Y), (255, 0, 0), cv2.FILLED) # preenchimento da barra
        cv2.putText(img, f'{int(volume_percentage_ui)} %', (40, VOL_BAR_MIN_Y + 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3) # texto da percentagem de volume do gesto
        
        # mostra volume atual do sistema
        current_os_volume_scalar = get_volume_os() 
        current_os_volume_percentage = int(current_os_volume_scalar * 100) 
        cv2.putText(img, f'Sys Vol: {current_os_volume_percentage}%', (WCAM_WIDTH - 250, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, volume_indicator_color_ui, 3) # texto do volume atual do sistema

        # calcula e exibe fps
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)

        cv2.imshow("Volume Control Hand Gesture", img) # nome da janela principal
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # pressionar 'q' para sair
            print("Info: 'q' pressed, exiting...")    
            break
finally:
    # liberta recursos ao terminar programa
    print("Info: Releasing resources...") 
    if platform.system() == "Linux" and pulse_instance_ref and pulse_instance_ref.connected:
        print("Info: Closing pulsectl connection.") 
        pulse_instance_ref.close()
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("Info: Program terminated gracefully.") 