import cv2
import mediapipe as mp
import numpy as np
import os
import time
import json
import sys # para argumentos de linha de comando
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image as keras_image # renomeado para evitar conflito com mp.image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# diretórios e caminhos para armazenamento de dados e modelos.

# diretório base do script atual
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # ex: /caminho/para/script_dir

# raiz do projeto (strg), dois níveis acima de src/main
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')) # ex: /caminho/para/projeto_strg

# dataset_dir: local dos dados de gestos (json).
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
# model_dir: local dos modelos treinados e ficheiros associados.
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "custom")
# faces_dataset_dir: diretório para armazenar imagens de facede linha de comandolhidas.
FACES_DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset_faces")
# known_faces_file: ficheiro para armazenar as features das faces conhecidas.
KNOWN_FACES_FILE = os.path.join(MODEL_DIR, "known_faces_data.npz")


# model_name: nome do ficheiro do modelo keras para gestos.
MODEL_NAME = "custom_gesture_model.keras"
# classes_path: caminho para o ficheiro json que mapeia os índices das classes para os nomes dos gestos.
CLASSES_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_classes.json")

# constantes para reconhecimento facial
FACE_IMG_SIZE = (224, 224) # tamanho esperado por mobilenetv2
FACE_RECOGNITION_THRESHOLD = 0.6 # limiar de similaridade para reconhecimento

# cria os diretórios necessários se não existirem.
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FACES_DATASET_DIR, exist_ok=True)

# inicialização dos módulos do mediapipe.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection

# modelos globais (inicializados quando necessário ou uma vez)
face_detection_model_mp = None # mediapipe face detection
feature_extractor_model_keras = None # keras mobilenetv2 for feature extraction

# função para inicializar modelos globais (detecção facial, extração de features)
def initialize_global_models():
    global face_detection_model_mp, feature_extractor_model_keras
    if face_detection_model_mp is None:
        face_detection_model_mp = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    if feature_extractor_model_keras is None:
        # usar mobilenetv2 pré-treinada no imagenet, sem a camada de classificação (include_top=false)
        # pooling='avg' adiciona uma camada globalaveragepooling2d no final
        feature_extractor_model_keras = MobileNetV2(weights='imagenet', include_top=False, input_shape=(FACE_IMG_SIZE[0], FACE_IMG_SIZE[1], 3), pooling='avg')
        print("MobileNetV2 feature extractor loaded.")

# utilitário para limpar o ecrã do terminal.
def clear():
    os.system('clear' if os.name == 'posix' else 'cls')

# menu do sistema de gestos.
def menu_gestos_dedicado(): 
    clear()
    print("""
# Gesture Recognition System
1 - Collect gesture samples
2 - Train custom gesture model
3 - Recognize gestures (uses trained model if available)
4 - Remove a specific gesture
q - Quit
""")
    return input("Choose an option: ").strip()

# menu do sistema de faces.
def menu_faces():
    clear()
    print("""
# Face System
1 - Enroll New Face
2 - Recognize Faces
0 - Back to Main Menu
""")
    return input("Choose an option: ").strip()

# permite escolher um gesto existente ou criar um novo.
def escolher_gesto(gestures):
    print("\\nAvailable gestures:")
    for idx, g in enumerate(gestures):
        print(f"{idx+1} - {g}")
    print("n - New gesture")
    op = input("Choose gesture (number or 'n'): ").strip()
    
    if op == 'n': # opção para novo gesto.
        nome = input("Enter name for the new gesture: ").strip()
        if nome: # nome fornecido, retorna.
            return nome
        else: # nome inválido, tenta novamente.
            print("Invalid name. Please try again.")
            return escolher_gesto(gestures)
    try:
        idx = int(op) - 1 # converte para índice base 0.
        if 0 <= idx < len(gestures): # índice válido.
            return gestures[idx]
    except ValueError: # entrada não numérica (e não 'n').
        pass # continua para mensagem de erro.
        
    print("Invalid option.")
    return escolher_gesto(gestures) # opção inválida, tenta novamente.

# coleta amostras de gestos.
def coletar_amostras():
    # carrega gestos existentes do dataset_dir
    existing_gestures_from_files = set()
    for f_name in os.listdir(DATASET_DIR):
        if f_name.endswith('.json'):
            # o nome do gesto é derivado do nome do ficheiro
            gesture_name_from_file = f_name[:-5] # remove .json
            existing_gestures_from_files.add(gesture_name_from_file)

    gestures = sorted(list(existing_gestures_from_files))
    
    print("Instructions: Press 'q' to quit, 's' to save, SPACE to start/stop recording.")
    print("Tip: Record at least 100 samples per gesture for better results.")
    
    cap = cv2.VideoCapture(0) # inicializa captura da webcam.
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # inicializa mediapipe hands para uma mão.
    mp_hands_model = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    current_gesture = escolher_gesto(gestures) # utilizador escolhe/cria gesto.
    if current_gesture is None: # nenhum gesto selecionado.
        cap.release()
        mp_hands_model.close()
        print("No gesture selected. Exiting collection.")
        return

    collected = [] # armazena landmarks da mão para o gesto.
    recording = False # controla o estado da gravação.
    frame_count = 0 # contador de frames gravados.
    last_saved_time = time.time() # para controlo de fps na gravação.
    fps_limit = 10 # frames por segundo para gravação.
    
    while True:
        ret, frame = cap.read() # lê frame da webcam.
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # converte para rgb (usado por mediapipe).
        results = mp_hands_model.process(frame_rgb) # deteta mãos no frame.
        annotated_frame = frame.copy() # cópia para desenhar anotações.
        height, width, _ = frame.shape # dimensões do frame.

        # desenha landmarks da mão, se detetados.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # painel de informações no ecrã.
        cv2.rectangle(annotated_frame, (0,0), (width, 80), (0,0,0), -1) # fundo preto para painel info.
        cv2.putText(annotated_frame, f"Gesture: {current_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2) # cyan
        cv2.putText(annotated_frame, f"Frames collected: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2) # verde
        
        status_text = "RECORDING" if recording else "PAUSED"
        status_color = (0,0,255) if recording else (255,255,255) # vermelho (gravando), branco (pausado).
        cv2.putText(annotated_frame, f"Status: {status_text}", (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # instruções na base do ecrã.
        cv2.putText(annotated_frame, "[SPACE] Record/Pause | [s] Save | [q] Quit", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # grava landmarks se gravação ativa e mão detetada.
        if results.multi_hand_landmarks and recording:
            now = time.time()
            # controlo de fps para gravação.
            if now - last_saved_time >= 1.0 / fps_limit:
                last_saved_time = now
                hand_data = [] # landmarks da mão atual.
                # extrai coordenadas (x, y, z) dos landmarks.
                # coordenadas normalizadas (0.0 a 1.0).
                for lm in results.multi_hand_landmarks[0].landmark:
                    hand_data.append({'x': lm.x, 'y': lm.y, 'z': lm.z}) # z: profundidade (z=0 na pulseira).
                collected.append(hand_data)
                frame_count += 1
        
        cv2.imshow("Gesture Collection", annotated_frame) # mostra frame anotado.
        key = cv2.waitKey(5) & 0xFF # espera tecla (timeout 5ms).

        if key == ord('q'): # 'q' para sair.
            break
        elif key == ord(' '): # espaço para gravar/pausar.
            recording = not recording
        elif key == ord('s'): # 's' para salvar dados.
            if collected:
                # nome do ficheiro json para o gesto.
                filename = os.path.join(DATASET_DIR, f"{current_gesture}.json")
                
                # carrega dados existentes para adicionar novos.
                if os.path.exists(filename):
                    try:
                        with open(filename, 'r') as f_content:
                            data_in_file = json.load(f_content)
                    except json.JSONDecodeError:
                        print(f"Warning: File {filename} is corrupted. Starting fresh for this gesture.")
                        data_in_file = {} # se o ficheiro estiver corrompido, começa de novo.
                else:
                    data_in_file = {} # se o ficheiro não existir, cria um dicionário vazio.

                # garante que a chave do gesto atual existe no dicionário.
                if current_gesture not in data_in_file:
                    data_in_file[current_gesture] = []
                
                # adiciona os dados coletados à lista existente para este gesto.
                data_in_file[current_gesture].extend(collected)
                
                # salva o dicionário atualizado de volta no ficheiro json.
                with open(filename, 'w') as f:
                    json.dump(data_in_file, f, indent=4) # indent=4 para melhor legibilidade do json.
                
                print(f"Saved {len(collected)} new frames for '{current_gesture}' to {filename}. Total frames for this gesture in file: {len(data_in_file[current_gesture])}")
                collected = [] # limpa a lista de coletados após salvar.
                frame_count = 0 # reinicia a contagem de frames.
            else:
                print("No new frames collected to save.")
                
    # liberta a câmara e fecha as janelas do opencv.
    cap.release()
    cv2.destroyAllWindows()
    mp_hands_model.close() # fecha o modelo mediapipe hands.

# remove um gesto específico e o modelo treinado associado.
def remover_gesto_individual():
    clear()
    print("# Remover Gesto Específico")

    # carrega gestos existentes do dataset_dir
    available_gestures = []
    for f_name in os.listdir(DATASET_DIR):
        if f_name.endswith('.json'):
            gesture_name = f_name[:-5] # remove .json
            available_gestures.append(gesture_name)
    
    available_gestures.sort()

    if not available_gestures:
        print("Nenhum gesto disponível para remover.")
        input("Pressione Enter para voltar ao menu...")
        return

    print("\\nGestos disponíveis para remoção:")
    for idx, name in enumerate(available_gestures):
        print(f"{idx + 1} - {name}")
    print("0 - Cancelar")

    while True:
        try:
            choice_str = input("Escolha o número do gesto a remover (ou 0 para cancelar): ").strip()
            choice = int(choice_str)
            if 0 <= choice <= len(available_gestures):
                break
            else:
                print("Opção inválida. Por favor, tente novamente.")
        except ValueError:
            print("Entrada inválida. Por favor, insira um número.")

    if choice == 0:
        print("Operação cancelada.")
        input("Pressione Enter para voltar ao menu...")
        return

    selected_gesture_name = available_gestures[choice - 1]
    
    confirm = input(f"Tem a certeza que quer remover o gesto '{selected_gesture_name}'? \\nIsto também irá remover o modelo treinado atual, se existir. (sim/não): ").strip().lower()

    if confirm == 'sim':
        gesture_file_path = os.path.join(DATASET_DIR, f"{selected_gesture_name}.json")
        try:
            if os.path.exists(gesture_file_path):
                os.remove(gesture_file_path)
                print(f"Ficheiro de dados do gesto '{selected_gesture_name}' removido: {gesture_file_path}")
            else:
                print(f"Ficheiro de dados do gesto '{selected_gesture_name}' não encontrado.")

            # remove o modelo e o ficheiro de classes, pois tornaram-se inválidos
            model_removed = False
            if os.path.exists(MODEL_PATH): # MODEL_PATH é usado aqui
                try:
                    os.remove(MODEL_PATH)
                    print(f"Modelo treinado removido: {MODEL_PATH}")
                    model_removed = True
                except OSError as e:
                    print(f"Erro ao remover o modelo {MODEL_PATH}: {e}")
            
            classes_removed = False
            if os.path.exists(CLASSES_PATH):
                try:
                    os.remove(CLASSES_PATH)
                    print(f"Ficheiro de classes do modelo removido: {CLASSES_PATH}")
                    classes_removed = True
                except OSError as e:
                    print(f"Erro ao remover o ficheiro de classes {CLASSES_PATH}: {e}")
            
            if model_removed or classes_removed:
                print("O modelo treinado foi removido porque o conjunto de dados de gestos foi alterado.")
                print("Por favor, treine o modelo novamente após fazer as alterações desejadas aos gestos.")
            print(f"Remoção do gesto '{selected_gesture_name}' processada.")

        except OSError as e:
            print(f"Erro ao remover o ficheiro do gesto {gesture_file_path}: {e}")
    else:
        print("Operação cancelada.")
    
    input("Pressione Enter para voltar ao menu...")

# ====== funções de reconhecimento facial ======

def preprocess_input_image_for_feature_extraction(face_image_bgr):
    # converte para rgb, redimensiona para o tamanho esperado pelo mobilenetv2, converte para array e pré-processa
    face_image_rgb = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2RGB)
    face_image_resized = cv2.resize(face_image_rgb, FACE_IMG_SIZE)
    img_array = keras_image.img_to_array(face_image_resized)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return mobilenet_preprocess_input(img_array_expanded)

def extract_face_features(processed_image_batch):
    global feature_extractor_model_keras
    if feature_extractor_model_keras is None:
        initialize_global_models() # garante que o modelo está carregado
    features = feature_extractor_model_keras.predict(processed_image_batch, verbose=0)
    return features.flatten() # retorna um array 1d de features

def enroll_face():
    clear()
    initialize_global_models() # garante que face_detection_model_mp está carregado
    print("# Enroll New Face")
    person_name = input("Enter the name of the person: ").strip()
    if not person_name:
        print("Name cannot be empty. Enrollment cancelled.")
        input("Press Enter to return...")
        return

    person_dir = os.path.join(FACES_DATASET_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("\\nPosition your face in the frame. Press SPACE to capture an image (up to 5 images).")
    print("Press 'q' to finish enrollment for this person.")
    
    captured_images = 0
    max_images_per_person = 5
    # flag para indicar se o utilizador quer sair do registo
    user_quit_enrollment = False 

    while captured_images < max_images_per_person and not user_quit_enrollment:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        display_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection_model_mp.process(frame_rgb)
        
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q'):
            user_quit_enrollment = True
            break

        if results.detections:
            # processa apenas a primeira face detetada para simplificar
            detection = results.detections[0] 
            mp_drawing.draw_detection(display_frame, detection)
            
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            xmin = int(bboxC.xmin * iw)
            ymin = int(bboxC.ymin * ih)
            width = int(bboxC.width * iw)
            height = int(bboxC.height * ih)
            
            margin = 20 
            face_xmin = max(0, xmin - margin)
            face_ymin = max(0, ymin - margin)
            face_xmax = min(iw, xmin + width + margin)
            face_ymax = min(ih, ymin + height + margin)

            cv2.rectangle(display_frame, (xmin, ymin), (xmin + width, ymin + height), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Captures: {captured_images}/{max_images_per_person}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if face_xmax > face_xmin and face_ymax > face_ymin:
                cropped_face = frame[face_ymin:face_ymax, face_xmin:face_xmax]
                if key_pressed == ord(' '):
                    if cropped_face.size > 0:
                        img_path = os.path.join(person_dir, f"{person_name}_{captured_images}.jpg")
                        cv2.imwrite(img_path, cropped_face)
                        print(f"Saved image {captured_images + 1} for {person_name} to {img_path}")
                        captured_images += 1
                        cv2.putText(display_frame, "CAPTURED!", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                    else:
                        print("Failed to capture valid face image (empty crop). Try adjusting position.")
        else:
             cv2.putText(display_frame, f"Captures: {captured_images}/{max_images_per_person}. Point to face.", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Enroll Face - Press SPACE to capture, 'q' to finish", display_frame)
        
        # sair se máximo de imagens atingido ou utilizador pressionar 'q'
        if captured_images >= max_images_per_person:
            print("Maximum images captured for this person.")
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if captured_images > 0:
        print(f"Enrollment for {person_name} complete. {captured_images} images saved.")
        # apaga ficheiro de features conhecidas para forçar regeneração
        if os.path.exists(KNOWN_FACES_FILE):
            try:
                os.remove(KNOWN_FACES_FILE)
                print(f"Removed existing known faces data file ({KNOWN_FACES_FILE}) to ensure regeneration.")
            except OSError as e:
                print(f"Could not remove {KNOWN_FACES_FILE}: {e}")
    else:
        print(f"No images captured for {person_name}. Enrollment incomplete.")
    
    input("Press Enter to return to the face menu...")

def load_or_generate_known_faces_data():
    global feature_extractor_model_keras
    if feature_extractor_model_keras is None:
        initialize_global_models()

    if os.path.exists(KNOWN_FACES_FILE):
        try:
            data = np.load(KNOWN_FACES_FILE, allow_pickle=True)
            known_face_features_list = data['features']
            known_face_names_list = data['names']
            print(f"Loaded {len(known_face_names_list)} known faces from {KNOWN_FACES_FILE}")
            return list(known_face_features_list), list(known_face_names_list)
        except Exception as e:
            print(f"Error loading {KNOWN_FACES_FILE}: {e}. Will regenerate.")

    known_face_features_list = []
    known_face_names_list = []
    
    print("Generating known faces data...")
    subdirs = [d for d in os.listdir(FACES_DATASET_DIR) if os.path.isdir(os.path.join(FACES_DATASET_DIR, d))]

    for person_name in subdirs:
        person_dir = os.path.join(FACES_DATASET_DIR, person_name)
        image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        person_features_sum = None
        num_person_images = 0

        for img_file in image_files:
            img_path = os.path.join(person_dir, img_file)
            face_image_bgr = cv2.imread(img_path)
            if face_image_bgr is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            
            # pré-processa e extrai features
            processed_batch = preprocess_input_image_for_feature_extraction(face_image_bgr)
            features = extract_face_features(processed_batch)
            
            if person_features_sum is None:
                person_features_sum = features
            else:
                person_features_sum += features
            num_person_images +=1
        
        if num_person_images > 0:
            # usar a média das features das imagens da pessoa
            avg_person_features = person_features_sum / num_person_images
            known_face_features_list.append(avg_person_features)
            known_face_names_list.append(person_name)
            print(f"Processed {num_person_images} images for {person_name}.")
        else:
            print(f"No valid images found for {person_name} in {person_dir}")


    if known_face_features_list:
        try:
            np.savez_compressed(KNOWN_FACES_FILE, features=np.array(known_face_features_list), names=np.array(known_face_names_list))
            print(f"Saved known faces data to {KNOWN_FACES_FILE}")
        except Exception as e:
            print(f"Error saving known faces data to {KNOWN_FACES_FILE}: {e}")
    else:
        print("No faces found to generate known faces data.")
            
    return known_face_features_list, known_face_names_list

# ====== fim das funções de reconhecimento facial ======

# treina o modelo de reconhecimento de gestos.
def treinar_modelo():
    print("Starting model training...")
    # carrega todos os dados de gestos do dataset_dir.
    all_data = []
    gesture_names = set()
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(DATASET_DIR, filename), 'r') as f:
                    data = json.load(f)
                    all_data.append(data)
                    gesture_names.update(data.keys()) # assume que as chaves do json são os nomes dos gestos.
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from file {filename} during training.")
            except Exception as e:
                print(f"Warning: Could not read gestures from file {filename} during training: {e}")

    if not all_data:
        print("No data found in dataset. Please collect samples first.")
        input("Press Enter to return to menu...")
        return

    # cria um mapeamento de nome de gesto para índice numérico.
    # ordena os nomes para garantir consistência (importante se o modelo for recarregado)
    sorted_gesture_names = sorted(list(gesture_names))
    gesture_map = {name: idx for idx, name in enumerate(sorted_gesture_names)}
    
    # prepara os dados para o modelo: x (features) e y (labels).
    X, y = [], []
    for data_file_content in all_data:
        for gesture_name, gesture_samples in data_file_content.items():
            if gesture_name not in gesture_map: # salvaguarda, embora deva estar pelo update anterior.
                print(f"Warning: Gesture '{gesture_name}' found in data but not in initial gesture_names set. Skipping.")
                continue
            for hand_landmarks_sample in gesture_samples:
                features = []
                for lm in hand_landmarks_sample:
                    features.extend([lm['x'], lm['y'], lm['z']])
                X.append(features)
                y.append(gesture_map[gesture_name])
    
    if not X or not y:
        print("No valid features or labels could be extracted from the dataset.")
        input("Press Enter to return to menu...")
        return

    X = np.array(X)
    y = np.array(y)

    # verifica se há dados suficientes após o processamento.
    if X.shape[0] == 0:
        print("No data available for training after processing.")
        input("Press Enter to return to menu...")
        return
    
    num_classes = len(gesture_map)
    if num_classes < 2 and X.shape[0] > 0: # stratify precisa de pelo menos 2 classes, ou y=none
        print(f"Only {num_classes} class(es) found. Training may not be effective. Need at least 2 for stratified split.")
        # para treino com uma classe, ou poucas amostras, a divisão e avaliação precisam ser manuseadas com cuidado.
        # pode-se optar por não dividir, ou usar uma divisão simples sem stratify.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)
    elif X.shape[0] > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else: # caso x seja vazio após tudo
        print("No data to train on after splitting.")
        input("Press Enter to return to menu...")
        return


    # define a arquitetura do modelo sequencial.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.3), # dropout para regularização.
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax') # camada de saída com ativação softmax para classificação.
    ])

    # compila o modelo.
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # define um callback para early stopping para evitar overfitting.
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, min_delta=0.001, restore_best_weights=True)
    
    print("Training gesture model...")
    # treina o modelo.
    history = model.fit(X_train, y_train, 
                        epochs=150,  # número aumentado de épocas, com early stopping.
                        batch_size=32, 
                        validation_data=(X_test, y_test) if X_test.size > 0 else None, # apenas se houver dados de teste
                        callbacks=[early_stopping],
                        verbose=1) # mostra o progresso do treino.

    # avalia o modelo no conjunto de teste.
    if X_test.size > 0 and y_test.size > 0:
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Gesture model test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
    else:
        print("No test data to evaluate gesture model.")

    # salva o modelo treinado e o mapeamento de classes.
    model.save(MODEL_PATH) # MODEL_PATH é usado aqui
    # guarda o mapeamento de índice para nome de gesto, que é o inverso de gesture_map.
    class_idx_to_name = {str(v): k for k, v in gesture_map.items()}
    with open(CLASSES_PATH, 'w') as f:
        json.dump(class_idx_to_name, f, indent=4)
    
    # plota e salva o histórico de treino (acurácia e perda).
    if history and history.history:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Gesture Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Gesture Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        plt.tight_layout()
        try:
            plt.savefig(os.path.join(MODEL_DIR, 'gesture_training_history.png'))
            plt.close() # fecha a figura para libertar memória
        except Exception as e:
            print(f"Error saving gesture training history plot: {e}")

    print("Gesture model trained and saved successfully!")
    input("Press Enter to return to menu...")


def reconhecer_gestos(): # renomeado de volta e removida lógica de face
    clear()
    print("Starting real-time gesture recognition...")
    
    # carregar modelo de gestos personalizado
    custom_gesture_model = None
    gesture_class_mapping = None 
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH): # MODEL_PATH é usado aqui
        try:
            custom_gesture_model = tf.keras.models.load_model(MODEL_PATH) # MODEL_PATH é usado aqui
            with open(CLASSES_PATH, 'r') as f:
                gesture_class_mapping = json.load(f) 
            print(f"Custom gesture model loaded: {len(gesture_class_mapping)} gestures.")
        except Exception as e:
            print(f"Error loading custom gesture model or class mapping: {e}.")
            custom_gesture_model = None
    else:
        print("Custom gesture model not found. Gesture recognition will be limited.")

    # inicializa a captura de vídeo.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # inicializa o mediapipe hands para deteção de landmarks de mãos.
    mp_hands_model = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # mediapipe default gesture recognizer (se o modelo .task existir)
    mp_gesture_recognizer = None
    gesture_recognizer_model_path = "gesture_recognizer.task" 
    if hasattr(mp, 'tasks') and hasattr(mp.tasks, 'vision') and os.path.exists(gesture_recognizer_model_path) and not custom_gesture_model:
        try:
            BaseOptions = mp.tasks.BaseOptions
            GestureRecognizer = mp.tasks.vision.GestureRecognizer
            GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            
            options = GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=gesture_recognizer_model_path),
                running_mode=VisionRunningMode.VIDEO, 
                num_hands=1 
            )
            mp_gesture_recognizer = GestureRecognizer.create_from_options(options)
            print("MediaPipe default gesture recognizer loaded as fallback.")
        except Exception as e:
            print(f"Error loading MediaPipe default gesture recognizer: {e}")
            mp_gesture_recognizer = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        annotated_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape # adicionado para ter dimensões do frame
        
        # ===== reconhecimento de gestos =====
        gesture_text = "-" 
        hand_results = mp_hands_model.process(frame_rgb) 
        if hand_results.multi_hand_landmarks:
            for single_hand_landmarks in hand_results.multi_hand_landmarks: 
                mp_drawing.draw_landmarks(
                    annotated_frame, single_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            if custom_gesture_model and gesture_class_mapping:
                hand_landmarks_for_model = hand_results.multi_hand_landmarks[0] 
                features = []
                for lm in hand_landmarks_for_model.landmark:
                    features.extend([lm.x, lm.y, lm.z])
                
                X_pred = np.array([features])
                predictions = custom_gesture_model.predict(X_pred, verbose=0)[0]
                predicted_idx = int(np.argmax(predictions))
                confidence = predictions[predicted_idx]
                
                recognition_threshold = 0.70 
                if confidence > recognition_threshold:
                    gesture_text = f"{gesture_class_mapping.get(str(predicted_idx), '? g ?')} ({confidence:.2f})"
                else:
                    gesture_text = f"Uncertain G ({confidence:.2f})"
            
            elif mp_gesture_recognizer: 
                mp_image_hands = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp_ms = int(time.time() * 1000)
                try:
                    recognition_result_hands = mp_gesture_recognizer.recognize_for_video(mp_image_hands, timestamp_ms)
                    if recognition_result_hands.gestures and recognition_result_hands.gestures[0]:
                        gesture_text = recognition_result_hands.gestures[0][0].category_name + " (MP Default)"
                except Exception as e:
                    gesture_text = "mp g-rec err"

        # exibe informações no ecrã
        cv2.rectangle(annotated_frame, (0, 0), (width, 60), (0,0,0), -1) # fundo para texto do gesto
        cv2.putText(annotated_frame, f"Gesture: {gesture_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2) # ciano
        
        cv2.putText(annotated_frame, "[q] Quit", (width-150, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        cv2.imshow("Gesture Recognition", annotated_frame)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    mp_hands_model.close()
    if mp_gesture_recognizer:
        mp_gesture_recognizer.close()

    print("Gesture recognition stopped.")
    input("Press Enter to return to menu...")


def reconhecer_faces(): # nova função apenas para reconhecimento facial
    clear()
    initialize_global_models() # garante que face_detection_model_mp e feature_extractor_model_keras estão carregados
    print("Starting real-time face recognition...")

    known_face_features, known_face_names = load_or_generate_known_faces_data()
    if not known_face_names:
        print("No known faces enrolled. Please enroll faces first.")
        input("Press Enter to return to menu...")
        return
    else:
        print(f"Loaded {len(known_face_names)} known face(s) for recognition.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # estado para suavização do nome exibido no painel
    panel_display_name_state = {"name": "Unknown", "persistence": 0}
    MAX_PERSISTENCE_FRAMES = 7 # nº de frames para manter um nome se não confirmado, mas face presente

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        annotated_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape

        # nome bruto das deteções no frame atual
        raw_name_from_current_frame_detections = "Unknown"
        any_face_detected_in_current_frame = False

        face_results_mp = face_detection_model_mp.process(frame_rgb)

        if face_results_mp.detections:
            any_face_detected_in_current_frame = True
            for detection in face_results_mp.detections:
                mp_drawing.draw_detection(annotated_frame, detection)
                identified_name_for_this_box = "Unknown" # nome para esta caixa de deteção
                
                bboxC = detection.location_data.relative_bounding_box
                xmin_det = int(bboxC.xmin * width)
                ymin_det = int(bboxC.ymin * height)
                w_det = int(bboxC.width * width)
                h_det = int(bboxC.height * height)
                
                margin = 20 
                face_xmin_crop = max(0, xmin_det - margin)
                face_ymin_crop = max(0, ymin_det - margin)
                face_xmax_crop = min(width, xmin_det + w_det + margin)
                face_ymax_crop = min(height, ymin_det + h_det + margin)

                if face_xmax_crop > face_xmin_crop and face_ymax_crop > face_ymin_crop:
                    current_face_bgr = frame[face_ymin_crop:face_ymax_crop, face_xmin_crop:face_xmax_crop]
                    
                    if current_face_bgr.size > 0:
                        processed_face_batch = preprocess_input_image_for_feature_extraction(current_face_bgr)
                        current_face_features = extract_face_features(processed_face_batch)
                        
                        if known_face_features: 
                            similarities = cosine_similarity(current_face_features.reshape(1, -1), np.array(known_face_features))
                            best_match_idx = np.argmax(similarities)
                            
                            if similarities[0, best_match_idx] > FACE_RECOGNITION_THRESHOLD:
                                identified_name_for_this_box = known_face_names[best_match_idx]
                                # atualiza candidato bruto se nome conhecido
                                # simplificação: último nome conhecido detetado no frame é o candidato
                                raw_name_from_current_frame_detections = identified_name_for_this_box
                        
                        # exibe nome instantâneo na caixa de deteção
                        cv2.putText(annotated_frame, identified_name_for_this_box, (xmin_det, ymin_det - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0) if identified_name_for_this_box != "Unknown" else (0,0,255), 2)
        
        # lógica de suavização para nome no painel superior
        if raw_name_from_current_frame_detections != "Unknown":
            panel_display_name_state["name"] = raw_name_from_current_frame_detections
            panel_display_name_state["persistence"] = 0
        else: # reconhecimento bruto do frame atual é "unknown"
            if any_face_detected_in_current_frame:
                if panel_display_name_state["name"] != "Unknown": # havia um nome estável conhecido
                    panel_display_name_state["persistence"] += 1
                    if panel_display_name_state["persistence"] >= MAX_PERSISTENCE_FRAMES:
                        panel_display_name_state["name"] = "Unknown" # perdeu o nome estável
                        panel_display_name_state["persistence"] = 0
                # se panel_display_name_state["name"] já era "unknown", continua "unknown"
            else: # nenhuma face detetada no frame
                panel_display_name_state["name"] = "Unknown"
                panel_display_name_state["persistence"] = 0

        # exibe nome da face (suavizado) no painel superior
        cv2.rectangle(annotated_frame, (0,0), (width, 60), (0,0,0), -1)
        cv2.putText(annotated_frame, f"Face: {panel_display_name_state['name']}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if panel_display_name_state['name'] != "Unknown" else (0,0,255), 2)
        cv2.putText(annotated_frame, "[q] Quit", (width-150, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        cv2.imshow("Face Recognition", annotated_frame)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    # modelos globais (face_detection_model_mp, feature_extractor_model_keras) não são fechados aqui
    # são reutilizáveis; fecho ideal na terminação da app ou quando objeto não é mais necessário.
    # simplificação: fechar câmara/janelas é suficiente para esta função.

    print("Face recognition stopped.")
    input("Press Enter to return to menu...")


def main():
    # verifica modo facial por argumento de linha de comando
    if len(sys.argv) > 1 and sys.argv[1] == '--face-mode':
        # loop do sistema de faces
        while True:
            clear()
            opcao_face = menu_faces()
            if opcao_face == '1':
                enroll_face()
            elif opcao_face == '2': 
                reconhecer_faces() 
            elif opcao_face == '0': # voltar ao menu principal (sair do modo face)
                print("Exiting Face System...")
                break
            else:
                print("Invalid option for face system. Please try again.")
                time.sleep(1)
    else:
        # loop do sistema de gestos (padrão)
        while True:
            clear()
            opcao_gesto = menu_gestos_dedicado()

            if opcao_gesto == '1':
                coletar_amostras()
            elif opcao_gesto == '2':
                treinar_modelo()
            elif opcao_gesto == '3':
                reconhecer_gestos()
            elif opcao_gesto == '4': 
                remover_gesto_individual()
            elif opcao_gesto == 'q':
                print("Exiting Gesture System...")
                break
            else:
                print("Invalid option. Please try again.")
                time.sleep(1)
    
    # limpeza final ao sair (modelos inicializados)
    global face_detection_model_mp
    if face_detection_model_mp:
        face_detection_model_mp.close()
        face_detection_model_mp = None 
        print("MediaPipe Face Detection model closed.")
    print("Application closed.")

if __name__ == "__main__":
    # ponto de entrada do script.
    main() 