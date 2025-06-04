import cv2
import numpy as np
import os
import pickle
from deepface import DeepFace
import mediapipe as mp
import time
import sys

# constantes para configuração
REF_DIR = "ref_faces" # diretório para imagens de referência de faces.
DATABASE_FILE = "face_database.pkl" # ficheiro da base de dados de faces (pickle).
STABILITY_THRESHOLD = 3 # nº de reconhecimentos consistentes para estabilizar nome.
DISTANCE_THRESHOLD = 0.4 # limiar de distância para deepface.verify considerar face verificada.
MAX_CAMERA_INDICES_TO_TRY = 3 # nº máximo de índices de câmara a tentar (0, 1, 2).
DEFAULT_FRAME_WIDTH = 640 # largura padrão do frame da câmara.
DEFAULT_FRAME_HEIGHT = 480 # altura padrão do frame da câmara.
RECOGNITION_INTERVAL_FRAMES = 30 # executar reconhecimento facial a cada n frames para otimizar.

print("Starting facial recognition program...")

# configuração de variáveis de ambiente para compatibilidade com OpenCV e Qt em alguns sistemas.
# os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0' # pode ser necessário em alguns sistemas windows.
# os.environ['QT_QPA_PLATFORM'] = 'xcb' # pode ser útil se correr em linux sem um display server completo ou com problemas de Qt.

# verifica e define variável de ambiente display se ausente (comum em headless/docker).
if 'DISPLAY' not in os.environ:
    print("Warning: DISPLAY environment variable not detected. Attempting to set to ':0'. GUI might not be available.")
    os.environ['DISPLAY'] = ':0'

# inicialização do mediapipe facemesh para deteção de landmarks faciais.
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils # utilitários de desenho do mediapipe.
# configura facemesh para detetar múltiplas faces (até 10) com confiança mínima.
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=10, min_detection_confidence=0.5)

print("MediaPipe FaceMesh initialized.")

# cria diretório de referência para imagens de faces, se não existir.
if not os.path.exists(REF_DIR):
    os.makedirs(REF_DIR)
    print(f"Directory '{REF_DIR}' created.")
else:
    print(f"Directory '{REF_DIR}' already exists.")

# carrega base de dados de faces existente ou cria uma nova.
face_database = {} # dicionário para mapear nome para caminho da imagem.
if os.path.exists(DATABASE_FILE):
    try:
        with open(DATABASE_FILE, 'rb') as f:
            face_database = pickle.load(f)
        print(f"Face database loaded. Contains {len(face_database)} faces.")
    except Exception as e:
        print(f"Error loading face database '{DATABASE_FILE}': {e}")
else:
    print(f"No existing face database found. A new one will be created at '{DATABASE_FILE}'.")

# tenta encontrar e abrir uma câmara.
cap = None
camera_found = False
for cam_index in range(MAX_CAMERA_INDICES_TO_TRY):
    print(f"Attempting to open camera index {cam_index}...")
    temp_cap = cv2.VideoCapture(cam_index)
    if temp_cap.isOpened():
        cap = temp_cap
        camera_found = True
        print(f"Camera index {cam_index} opened successfully.")
        break
    else:
        print(f"Camera index {cam_index} not found or could not be opened.")
        temp_cap.release() # liberta objeto videocapture se não aberto corretamente.

if not camera_found:
    print(f"Error: Could not open any camera after trying indices 0 to {MAX_CAMERA_INDICES_TO_TRY-1}. Exiting.")
    # aqui poderia adicionar sys.exit(1) se a terminação imediata for desejada.
    # por agora, o programa continuará, mas o loop principal provavelmente falhará.
    # considerar uma melhor forma de lidar com isto dependendo do comportamento desejado.

# define resolução da câmara, se encontrada.
if cap:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_HEIGHT)

# variáveis de estado da aplicação.
counter = 0 # contador de frames processados.
training_mode = False # flag: modo de treino ativo.
current_name = "" # nome da pessoa a guardar (modo de treino).
matched_name = "Unknown" # nome da face reconhecida.

# variáveis para estabilização do reconhecimento facial.
recent_recognitions = [] # lista dos últimos n nomes reconhecidos.
current_stable_name = "Unknown" # nome estável após n reconhecimentos consistentes.

# guarda uma nova face na base de dados.
# frame: frame da câmara com a face a guardar.
# name: nome associado à face.
def save_face(frame, name):
    # guarda imagem da face no diretório de referência.
    img_path = os.path.join(REF_DIR, f"{name}.jpg")
    cv2.imwrite(img_path, frame)
    # atualiza base de dados em memória e guarda em ficheiro pickle.
    face_database[name] = img_path
    with open(DATABASE_FILE, 'wb') as f:
        pickle.dump(face_database, f)
    print(f"Face '{name}' saved to '{img_path}'. Database updated.")

# reconhece uma face no frame fornecido.
# frame: frame da câmara para análise.
# retorna: nome da face reconhecida (estável) ou "unknown".
def recognize_face(frame):
    global matched_name, recent_recognitions, current_stable_name
    
    if not face_database: # se base de dados vazia, nada a reconhecer.
        # print("Debug: recognize_face called with empty database.") # Removido print de depuração
        return "Unknown"
    
    highest_confidence_metric = 0 # para deepface, menor 'distance' é melhor (inicializado para lógica de <).
    best_match_name = "Unknown"
    
    # itera sobre todas as faces na base de dados.
    for name, img_path in face_database.items():
        try:
            if not os.path.exists(img_path):
                print(f"Warning: Image path '{img_path}' for name '{name}' not found in database. Skipping.")
                continue
            
            reference_img = cv2.imread(img_path)
            if reference_img is None:
                print(f"Warning: Could not read image from path '{img_path}' for name '{name}'. Skipping.")
                continue

            # usa deepface.verify para comparar frame atual com imagem de referência.
            # enforce_detection=false pois deteção já é feita por mediapipe.
            result = DeepFace.verify(frame, reference_img, model_name='VGG-Face', distance_metric='cosine', enforce_detection=False)
            
            # se face verificada e distância abaixo do limiar.
            if result['verified'] and result['distance'] < DISTANCE_THRESHOLD:
                # deepface retorna 'distance' (menor é melhor).
                # esta lógica procura a menor distância (match mais similar).
                # highest_confidence_metric armazena a menor distância encontrada.
                if best_match_name == "Unknown" or result['distance'] < highest_confidence_metric: # 
                    highest_confidence_metric = result['distance']
                    best_match_name = name
        except Exception as e:
            print(f"Error during DeepFace verification for '{name}': {e}")
            continue # continua para a próxima face.
    
    # lógica de estabilização do reconhecimento.
    recent_recognitions.append(best_match_name) # adiciona reconhecimento atual à lista.
    
    # mantém lista de reconhecimentos recentes com tamanho máximo (stability_threshold).
    if len(recent_recognitions) > STABILITY_THRESHOLD:
        recent_recognitions.pop(0) # remove o reconhecimento mais antigo.
    
    # se lista atingiu o tamanho do limiar de estabilidade.
    if len(recent_recognitions) == STABILITY_THRESHOLD:
        # verifica se todos os reconhecimentos recentes são iguais (e não "unknown").
        is_stable = all(name == recent_recognitions[0] for name in recent_recognitions)
        if is_stable and recent_recognitions[0] != "Unknown":
            current_stable_name = recent_recognitions[0]
        elif not is_stable: # se não estável, reseta para "unknown" até nova estabilização.
            current_stable_name = "Unknown"
    
    matched_name = current_stable_name # atualiza nome globalmente correspondido.
    return matched_name # retorna nome estável atual.

# deteta faces e desenha landmarks e nomes.
# frame: frame da câmara para processar e anotar.
def detect_faces(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # converte frame para rgb para mediapipe.
    results = face_mesh.process(rgb_frame) # processa frame com facemesh.
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # desenha contornos da face.
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1), # verde
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)) # verde
            
            h, w, _ = frame.shape # obtém dimensões do frame.
            
            # calcula bounding box da face a partir dos landmarks.
            x_min, x_max, y_min, y_max = w, 0, h, 0
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
            
            # calcula posição para texto do nome.
            # centraliza texto horizontalmente em relação à face.
            face_center_x = (x_min + x_max) // 2
            (text_width, text_height), baseline = cv2.getTextSize(matched_name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_x = face_center_x - text_width // 2
            
            # posiciona texto acima da bounding box da face.
            text_y = y_min - 20
            
            # ajusta posição y do texto para não sair do topo do ecrã.
            if text_y < text_height + 5: # adiciona pequena margem.
                text_y = text_height + 15 # move para baixo se perto do topo.

            # coordenadas para fundo do texto (melhor legibilidade).
            text_bg_x1 = max(0, text_x - 5)
            text_bg_y1 = max(0, text_y - text_height - baseline - 5)
            text_bg_x2 = min(w, text_x + text_width + 5)
            text_bg_y2 = min(h, text_y + baseline + 5)
            
            # desenha retângulo de fundo semi-transparente.
            overlay = frame.copy()
            cv2.rectangle(overlay, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (0, 0, 0), -1) # preto
            alpha = 0.6 # nivel de transparência.
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # desenha texto do nome da face (com contorno para destaque).
            cv2.putText(frame, matched_name, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)  # contorno preto
            cv2.putText(frame, matched_name, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # texto verde

# funções para análise de imagem via cli (dlib style)
def load_known_faces_cli(known_faces_dir):
    # aviso: esta é uma função placeholder.
    # a lógica original para carregar descritores dlib não foi encontrada.
    print(f"aviso: a função 'load_known_faces_cli' é um placeholder e precisa ser implementada para carregar descritores dlib de {known_faces_dir}")
    return [], [] # retorna descritores e nomes vazios

def analyze_image_cli(image_path, known_face_descriptors, known_face_names, threshold=0.6):
    # aviso: esta função requer inicialização de modelos dlib (não incluída aqui).
    # por exemplo:
    # import dlib
    # dlib_face_detector = dlib.get_frontal_face_detector() 
    # shape_predictor_path = "caminho/para/shape_predictor_68_face_landmarks.dat" # precisa ser definido
    # shape_predictor = dlib.shape_predictor(shape_predictor_path)
    # facerec_model_path = "caminho/para/dlib_face_recognition_resnet_model_v1.dat" # precisa ser definido
    # facerec = dlib.face_recognition_model_v1(facerec_model_path)

    print(f"aviso: 'analyze_image_cli' está a usar uma estrutura placeholder para processamento dlib.")
    
    # placeholder para carregamento de imagem e deteção inicial de faces
    # img = cv2.imread(image_path)
    # if img is None:
    #     print(f"erro: não foi possível carregar a imagem de {image_path}")
    #     return None
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # detected_faces = dlib_face_detector(img_rgb, 1) # 'dlib_face_detector' não está inicializado
    
    # o código abaixo é o loop de processamento de faces da tentativa anterior de refatoração.
    # assume que 'img_rgb', 'detected_faces', 'shape_predictor', 'facerec', 'np' estão disponíveis e configurados.
    # estas variáveis não estão definidas neste contexto placeholder.
    
    # o bloco seguinte é deixado com a lógica de refatoração de comentários aplicada,
    # mas não será funcional sem a configuração dlib completa.
    
    # simula um resultado para evitar mais erros; substitua pela lógica real.
    print(f"a analisar imagem {image_path} (simulação).")
    # img_bgr_placeholder = np.zeros((100,100,3), dtype=np.uint8) # cria uma imagem preta
    # cv2.putText(img_bgr_placeholder, "Placeholder", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    
    # o código original do loop dlib (comentado para evitar erros de execução imediata):
    # analisa cada rosto encontrado
    # for i, face_rect in enumerate(detected_faces): # 'detected_faces' indefinido
    #     # obtém as coordenadas do retângulo do rosto
    #     x1, y1, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()

    #     # extrai o descritor do rosto (um vetor de 128 dimensões)
    #     # 'shape' viria de shape_predictor(img_rgb, face_rect)
    #     face_descriptor = facerec.compute_face_descriptor(img_rgb, shape) # 'facerec', 'img_rgb', 'shape' indefinidos

    #     # compara o descritor do rosto atual com os descritores conhecidos
    #     distances = [np.linalg.norm(np.array(face_descriptor) - np.array(known_descriptor)) for known_descriptor in known_face_descriptors] # 'np' indefinido
    #     min_distance_idx = np.argmin(distances)
    #     min_distance = distances[min_distance_idx]

    #     # se a distância mínima for menor que o limiar, o rosto é reconhecido
    #     if min_distance < threshold:
    #         name = known_face_names[min_distance_idx]
    #         # desenha um retângulo verde e o nome da pessoa
    #         cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.putText(img_rgb, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    #     else:
    #         # se o rosto não for reconhecido, desenha um retângulo vermelho e "desconhecido"
    #         cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #         cv2.putText(img_rgb, "desconhecido", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # converte a imagem de volta para bgr para exibição com opencv
    # img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) # 'img_rgb' indefinido
    # return img_bgr
    
    # retorna uma imagem placeholder para que o fluxo __main__ não quebre visualmente.
    img_placeholder = np.zeros((DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(img_placeholder, f"Placeholder for {os.path.basename(image_path)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(img_placeholder, "DLIB CLI analysis needs full implementation.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2)
    return img_placeholder

# exibe teclas de atalho disponíveis.
print("\\nKeybinds:")
print("- T: Toggle Training mode")
print("- C: Capture face (only in training mode)")
print("- Q: Quit program")
print("---------------------------------------------------")

start_time_loop = time.time() # tempo de início do loop principal.
window_name = 'Facial Recognition STRG' # nome da janela de visualização.

# cria e redimensiona janela do opencv.
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT)

# tenta mostrar primeiro frame para verificar se gui funciona
# (ajuda a diagnosticar problemas de display).
if cap and cap.isOpened():
    ret_test, test_frame = cap.read()
    if ret_test:
        print("First frame captured successfully. Displaying...")
        cv2.imshow(window_name, test_frame)
        cv2.waitKey(100) # pequena pausa para renderizar janela.
    else:
        print("Error: Failed to capture the first test frame from camera.")
else:
    print("Warning: Camera not available or not opened. Cannot display initial test frame.")

# loop principal da aplicação.
while True:
    if not cap or not cap.isOpened():
        print("Error: Camera is not available or has been disconnected. Attempting to reconnect...")
        # tenta reabrir a câmara.
        # esta é uma tentativa simples, pode precisar de lógica mais robusta.
        time.sleep(1) # espera um pouco antes de tentar reabrir.
        temp_cap_reconnect = cv2.VideoCapture(0) # tenta o índice 0 por defeito.
        if temp_cap_reconnect.isOpened():
            cap = temp_cap_reconnect
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_HEIGHT)
            print("Camera reconnected successfully.")
        else:
            print("Error: Failed to reconnect camera. Exiting loop.")
            temp_cap_reconnect.release()
            break # sai do loop principal se não conseguir reconectar.
    try:
        # monitoriza se o loop está a correr mas sem processar frames (indicativo de problema).
        # current_time_loop = time.time()
        # if current_time_loop - start_time_loop > 5 and counter == 0 and cap.isOpened(): # removido, pois 'counter' é global e pode ser >0 de execuções anteriores
        #     print("Warning: Main loop running, but no frames processed recently.")
        #     print(f"Webcam status: {cap.isOpened()}")
        
        ret, frame = cap.read() # lê um frame da câmara.
        
        if not ret:
            print("Error: Failed to retrieve frame from camera. Retrying...")
            # não quebra imediatamente, tenta continuar ou reconectar na próxima iteração.
            time.sleep(0.1) # pequena pausa.
            continue 
        
        counter += 1 # incrementa o contador de frames.
        frame = cv2.flip(frame, 1) # inverte o frame horizontalmente (efeito espelho).
        
        # exibe informações do modo de treino no frame.
        if training_mode:
            cv2.putText(frame, "TRAINING MODE", (10, 70), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Amarelo
            cv2.putText(frame, f"Capturing for: {current_name}", (10, 100), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Amarelo
        
        # executa o reconhecimento facial a cada N frames para otimizar o desempenho.
        if counter % RECOGNITION_INTERVAL_FRAMES == 0 and not training_mode:
            try:
                # faz uma cópia do frame para o reconhecimento para evitar problemas com modificações.
                matched_name_temp = recognize_face(frame.copy())
                # print(f"Recognized: {matched_name_temp}") # Debug: print para ver o que foi reconhecido
            except Exception as e:
                print(f"Error during periodic face recognition: {e}")
        
        detect_faces(frame) # deteta faces e desenha informações no frame.
        
        cv2.imshow(window_name, frame) # exibe o frame processado.
        
        key = cv2.waitKey(1) & 0xFF # espera por uma tecla (1ms timeout).
        
        if key == ord('q'): # tecla 'q' para sair.
            print("Exiting program...")
            break
        
        elif key == ord('t'): # tecla 't' para alternar o modo de treino.
            training_mode = not training_mode
            if training_mode:
                print("\\nTraining Mode Activated.")
                name_input = input("Enter name for the person to capture: ").strip()
                if name_input: # verifica se algum nome foi inserido.
                    current_name = name_input
                    print(f"Now capturing for: {current_name}. Press 'c' to save face.")
                else:
                    print("No name entered. Exiting training mode.")
                    training_mode = False # sai do modo de treino se nenhum nome for fornecido.
            else:
                print("Training Mode Deactivated.")
        
        elif key == ord('c') and training_mode: # tecla 'c' para capturar e guardar uma face no modo de treino.
            if current_name: # certifica-se que um nome foi definido.
                save_face(frame.copy(), current_name)
                training_mode = False # desativa o modo de treino após guardar.
                print(f"Face for '{current_name}' saved. Training mode deactivated.")
            else:
                print("Cannot save face: No name specified. Activate training mode ('t') and enter a name first.")
    
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
        # adicionar aqui mais detalhes do erro se necessário, e.g., traceback.
        break # sai do loop em caso de erro não tratado.

# liberta a câmara e fecha todas as janelas do OpenCV ao sair.
if cap:
    cap.release()
cv2.destroyAllWindows()
print("Program terminated.")

# ponto de entrada principal do script (para análise de imagem cli)
if __name__ == "__main__":
    # verifica se o número correto de argumentos da linha de comando foi fornecido
    if len(sys.argv) > 1: # modificado para permitir execução sem args (cai no loop da webcam)
        if len(sys.argv) == 3:
            print("\\n# a executar em modo de análise de imagem cli...")
            # obtém os caminhos dos diretórios e da imagem a partir dos argumentos
            known_faces_dir_arg = sys.argv[1]
            image_path_to_analyze_arg = sys.argv[2]

            # carrega os descritores de rostos conhecidos
            known_face_descriptors_cli, known_face_names_cli = load_known_faces_cli(known_faces_dir_arg)
            # analisa a imagem
            output_image_cli = analyze_image_cli(image_path_to_analyze_arg, known_face_descriptors_cli, known_face_names_cli)

            # mostra a imagem resultante
            cv2.imshow("análise de imagem cli", output_image_cli)
            cv2.waitKey(0) # espera por uma tecla ser pressionada
            cv2.destroyAllWindows() # fecha todas as janelas do opencv
            print("# análise de imagem cli concluída.")
        else:
            print("utilização para cli: python main.py <caminho_para_diretorio_imagens_conhecidas> <caminho_para_imagem_a_analisar>")
            print("a iniciar modo webcam por defeito...")
            # (aqui o código entraria no loop da webcam se estivesse encapsulado numa função main_webcam())
            # como o loop da webcam está no escopo global abaixo deste if, ele será executado se não houver sys.exit()
    else:
        # se nenhum argumento cli for fornecido, assume-se que o loop da webcam (se presente no escopo global) deve correr.
        print("\\n# nenhum argumento cli fornecido. se o loop da webcam estiver presente, ele será executado.")
        # (o loop da webcam existente no ficheiro original continuará a ser executado após este bloco __main__)
        pass # permite que o script continue para o loop da webcam global, se existir
