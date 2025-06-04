import cv2
import mediapipe as mp
import numpy as np
import os
import time
import json
import math
# import datetime # não usado diretamente
import tensorflow as tf
# from tensorflow import keras # keras acedido via tf.keras
from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential # acedido via tf.keras.models.Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed # acedido via tf.keras.layers
# from tensorflow.keras.utils import to_categorical # acedido via tf.keras.utils
import matplotlib.pyplot as plt
# import sys # não necessário se SCRIPT_DIR for usado para caminhos relativos
import shutil # para remover diretórios recursivamente
from sklearn.metrics import classification_report, confusion_matrix

# obtém o diretório absoluto do script
# útil para caminhos relativos de dados e modelos, garante que o script funcione independentemente de onde é chamado.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# diretórios de dados e modelos, baseados na localização do script
# data_dir: armazena dados de sequências de palavras recolhidas (.npy)
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "word_recognition_data")
# model_dir: armazena modelo treinado e ficheiros associados (ex: mapeamento de classes)
MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "word_recognition_model")

# nome do ficheiro do modelo keras e caminho completo
MODEL_NAME = "word_recognition_model.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
# caminho para ficheiro json que mapeia índices de classes para nomes de palavras
CLASSES_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_classes.json")

# cria data_dir e model_dir se não existirem
# exist_ok=true evita erros se diretórios já existirem
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# inicialização de módulos mediapipe para deteção de mãos, corpo e desenho de landmarks
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose # mp_pose: deteção de landmarks do corpo

# configurações para coleta de sequências de palavras
SEQUENCE_LENGTH = 30  # sequence_length: nº de frames (pontos de dados temporais) por amostra
MIN_DETECTION_CONFIDENCE = 0.5 # min_detection_confidence: confiança mínima para deteção inicial de mãos/corpo
MIN_TRACKING_CONFIDENCE = 0.5  # min_tracking_confidence: confiança mínima para rastreamento subsequente
MAX_NUM_HANDS = 1 # max_num_hands: processa só uma mão para simplificar extração de features
# min_samples_per_class_for_split: mínimo de amostras por palavra para divisão treino/teste estratificada, evita erros
MIN_SAMPLES_PER_CLASS_FOR_SPLIT = 2 

# default_words: lista de palavras padrão para popular interface de coleta
DEFAULT_WORDS = ['ola', 'adeus', 'obrigado', 'sim', 'nao']

# utilitário para limpar o terminal
def clear():
    os.system('clear' if os.name == 'posix' else 'cls')

# menu principal da app de reconhecimento de palavras e retorna escolha do utilizador
def menu_principal():
    print("""
# Word Gesture Recognition System
1 - Collect word sequences
2 - Train word recognition model
3 - Recognize words (uses trained model if available)
4 - Remove word data
q - Quit
""")
    return input("Choose an option: ").strip()

# permite escolher palavra de lista existente ou criar nova para coleta de dados
# words: lista de nomes de palavras disponíveis
def escolher_palavra_para_coleta(words):
    print("\nAvailable words for collection:")
    for idx, word in enumerate(words):
        print(f"{idx+1} - {word}")
    print("n - New word")
    op = input("Choose word (number or 'n'): ").strip().lower()
    
    if op == 'n':
        nome = input("Enter name for the new word (e.g., 'eat', 'help'): ").strip().lower()
        # valida nome de palavra única, sem espaços, por simplicidade
        if nome and ' ' not in nome:
            return nome
        else:
            print("Invalid name. Please use a single word without spaces.")
            return escolher_palavra_para_coleta(words) # tenta novamente se nome inválido
    try:
        idx = int(op) - 1
        if 0 <= idx < len(words):
            return words[idx]
    except ValueError:
        pass # erro de conversão para int tratado pela mensagem de opção inválida abaixo
        
    print("Invalid option. Please try again.")
    return escolher_palavra_para_coleta(words) # tenta novamente se opção inválida

# auxiliar para calcular ponto médio entre dois pontos 3d (landmarks)
# p1, p2: landmarks mediapipe (com atributos x, y, z)
def _calcular_ponto_medio(p1, p2):
    return (p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2

# extrai e combina pontos chave normalizados de mãos e corpo a partir de resultados mediapipe
# results_hands: resultado da deteção de mãos mediapipe
# results_pose: resultado da deteção de pose (corpo) mediapipe
# retorna array numpy com pontos chave concatenados e tempo de execução da extração
def extrair_pontos_chave(results_hands, results_pose):
    start_time = time.time() # regista tempo de início para calcular duração
    pontos_chave = []
    
    # processamento de pontos da mão (21 landmarks, 3 coords cada = 63 features)
    num_hand_landmarks = 21
    # array de zeros para caso de mão não detetada
    mao_vazia = np.zeros(num_hand_landmarks * 3).tolist()

    if results_hands and results_hands.multi_hand_landmarks:
        # assume max_num_hands = 1, usa a primeira mão detetada
        hand_landmarks = results_hands.multi_hand_landmarks[0]
        wrist_lm = hand_landmarks.landmark[0] # pulso (landmark 0) como ponto de referência
        
        # normaliza pontos da mão subtraindo coordenadas do pulso
        # torna features mais robustas a translações da mão
        pontos_mao_normalizados = []
        for lm in hand_landmarks.landmark:
            pontos_mao_normalizados.extend([lm.x - wrist_lm.x, lm.y - wrist_lm.y, lm.z - wrist_lm.z])
        pontos_chave.extend(pontos_mao_normalizados)
    else:
        # se nenhuma mão detetada, adiciona zeros correspondentes
        pontos_chave.extend(mao_vazia)
    
    # processamento de pontos do corpo selecionados
    # Índices landmarks mediapipe pose: nose (0), left shoulder (11), right shoulder (12),
    # left elbow (13), right elbow (14), left wrist (15), right wrist (16), left hip (23), right hip (24).
    pose_indices_selecionados = [0, 11, 12, 13, 14, 15, 16, 23, 24]
    num_pose_landmarks_selecionados = len(pose_indices_selecionados)
    # array de zeros para caso de landmarks do corpo não detetados
    corpo_vazio = np.zeros(num_pose_landmarks_selecionados * 3).tolist()

    if results_pose and results_pose.pose_landmarks:
        all_pose_landmarks = results_pose.pose_landmarks.landmark
        
        # modificado: verifica se pelo menos os ombros estão visíveis para tentar normalização
        # idealmente, todos os 4 (ombros e ancas) deveriam estar visíveis
        left_shoulder_visible = idx_is_valid_and_visible(all_pose_landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value, MIN_DETECTION_CONFIDENCE)
        right_shoulder_visible = idx_is_valid_and_visible(all_pose_landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, MIN_DETECTION_CONFIDENCE)
        left_hip_visible = idx_is_valid_and_visible(all_pose_landmarks, mp_pose.PoseLandmark.LEFT_HIP.value, MIN_DETECTION_CONFIDENCE)
        right_hip_visible = idx_is_valid_and_visible(all_pose_landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value, MIN_DETECTION_CONFIDENCE) # min_detection_confidence aqui

        # if all(idx < len(all_pose_landmarks) and all_pose_landmarks[idx].visibility > MIN_DETECTION_CONFIDENCE 
        #        for idx in [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, 
        #                    mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value]):
        if left_shoulder_visible and right_shoulder_visible: # condição modificada
            
            ombro_e_lm = all_pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            ombro_d_lm = all_pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            centro_ombros_x, centro_ombros_y, centro_ombros_z = _calcular_ponto_medio(ombro_e_lm, ombro_d_lm)

            if left_hip_visible and right_hip_visible:
                anca_e_lm = all_pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                anca_d_lm = all_pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                centro_ancas_x, centro_ancas_y, centro_ancas_z = _calcular_ponto_medio(anca_e_lm, anca_d_lm)
                # ponto de referência final do torso usando ombros e ancas
                ref_x = (centro_ombros_x + centro_ancas_x) / 2
                ref_y = (centro_ombros_y + centro_ancas_y) / 2
                ref_z = (centro_ombros_z + centro_ancas_z) / 2
            else:
                # fallback: usa centro dos ombros como ponto de referência se ancas não bem detetadas
                print("Warning: Hips not clearly visible, using shoulder center for pose normalization reference.")
                ref_x, ref_y, ref_z = centro_ombros_x, centro_ombros_y, centro_ombros_z

            # normaliza landmarks do corpo selecionados em relação ao ponto de referência do torso
            # normalização adicional pela escala (ex: dividir pela distância entre ombros) poderia ser adicionada
            pontos_corpo_normalizados = []
            for idx in pose_indices_selecionados:
                if idx < len(all_pose_landmarks) and all_pose_landmarks[idx].visibility > MIN_DETECTION_CONFIDENCE:
                    lm = all_pose_landmarks[idx]
                    pontos_corpo_normalizados.extend([lm.x - ref_x, lm.y - ref_y, lm.z - ref_z])
                else: 
                    # se landmark específico não detetado ou baixa confiança, preenche com zeros (relativos a ref_point, zero após normalização)
                    pontos_corpo_normalizados.extend([0.0, 0.0, 0.0])
            pontos_chave.extend(pontos_corpo_normalizados)
        else: 
             # se landmarks de referência (ombros/ancas) não totalmente detetados ou visíveis
             # adicionado print de depuração para visibilidade de landmarks de referência
            debug_visibility_info = "Visibilities: "
            if results_pose and results_pose.pose_landmarks:
                lm_indices_ref = {
                    "L_Shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                    "R_Shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                    "L_Hip": mp_pose.PoseLandmark.LEFT_HIP.value,
                    "R_Hip": mp_pose.PoseLandmark.RIGHT_HIP.value
                }
                for name, idx in lm_indices_ref.items():
                    if idx < len(all_pose_landmarks):
                        debug_visibility_info += f"{name}: {all_pose_landmarks[idx].visibility:.2f} "
                    else:
                        debug_visibility_info += f"{name}: N/A "
            else:
                debug_visibility_info += "No pose landmarks detected."
            print(f"DEBUG: Body ref landmarks issue. {debug_visibility_info}")
            print("Warning: Body reference landmarks (shoulders/hips) not fully detected or low confidence. Using zeroed body landmarks.")
            pontos_chave.extend(corpo_vazio)
    else:
        # se nenhuma pose detetada, adiciona zeros correspondentes
        pontos_chave.extend(corpo_vazio)

    end_time = time.time() # regista tempo de fim
    execution_time = end_time - start_time # calcula tempo total de extração
    return np.array(pontos_chave), execution_time

# helper para verificar validade e visibilidade do landmark
def idx_is_valid_and_visible(landmarks_list, landmark_idx, min_visibility_threshold):
    return landmark_idx < len(landmarks_list) and landmarks_list[landmark_idx].visibility > min_visibility_threshold

# função principal para coleta de sequências de frames para cada palavra gestual
def coletar_sequencias_palavras():
    # carrega palavras já coletadas de nomes de diretórios em data_dir e combina com default_words
    palavras_coletadas = set(DEFAULT_WORDS)
    if os.path.exists(DATA_DIR):
        for entry in os.listdir(DATA_DIR):
            if os.path.isdir(os.path.join(DATA_DIR, entry)):
                palavras_coletadas.add(entry.lower()) # normaliza para minúsculas
    
    palavras_para_escolha = sorted(list(palavras_coletadas))

    # print("# Coleta de Sequências de Palavras Gestuais") # removido, título implícito
    print(f"Collecting {SEQUENCE_LENGTH} frames per word sample.")
    print("Instructions: Press 'q' to quit collection mode.")
    print("Press SPACE to start/stop recording ONE sample of the current word.")
    print("Once started, maintain the gesture/movement until frames are automatically collected.")

    cap = cv2.VideoCapture(0) # inicializa captura de vídeo
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # bloco 'with' para garantir que modelos mediapipe são fechados corretamente
    with mp_hands.Hands(max_num_hands=MAX_NUM_HANDS, 
                        min_detection_confidence=MIN_DETECTION_CONFIDENCE, 
                        min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as hands_model, \
         mp_pose.Pose(min_detection_confidence=MIN_DETECTION_CONFIDENCE, 
                      min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as pose_model:

        palavra_atual = escolher_palavra_para_coleta(palavras_para_escolha)
        if not palavra_atual: # se utilizador não escolher palavra (ex: cancela criação)
            cap.release()
            # cv2.destroyAllWindows() # não necessário aqui se loop principal não começar
            print("No word selected. Exiting collection mode.")
            return
        
        print(f"\nCollecting for word: '{palavra_atual}'")
        print("Prepare and press SPACE to record a sample.")

        coletando_amostra_atual = False # flag para controlar se gravação de amostra está ativa
        frames_desta_amostra = [] # lista para armazenar frames da amostra atual
        
        while cap.isOpened():
            ret, frame = cap.read() # lê frame da webcam
            if not ret:
                print("Error: Failed to capture frame from camera.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # converte para rgb
            frame_rgb.flags.writeable = False # otimização: marca frame como não gravável antes de processamento mediapipe
            results_hands = hands_model.process(frame_rgb)
            results_pose = pose_model.process(frame_rgb)
            frame_rgb.flags.writeable = True # reabilita escrita no frame
            
            # converte de volta para bgr para exibição com opencv
            annotated_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # desenha landmarks das mãos
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            
            # desenha landmarks do corpo
            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # lógica de coleta de frames para amostra atual
            if coletando_amostra_atual:
                pontos_chave_frame_atual, _ = extrair_pontos_chave(results_hands, results_pose)
                frames_desta_amostra.append(pontos_chave_frame_atual)
                # exibe o progresso da coleta no ecrã
                cv2.putText(annotated_frame, f"Recording... {len(frames_desta_amostra)}/{SEQUENCE_LENGTH}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) # Vermelho.

                # quando o número desejado de frames para a sequência é atingido.
                if len(frames_desta_amostra) == SEQUENCE_LENGTH:
                    caminho_palavra_dir = os.path.join(DATA_DIR, palavra_atual)
                    os.makedirs(caminho_palavra_dir, exist_ok=True)
                    
                    # determina o próximo número de sequência para o nome do ficheiro.
                    num_sequencias_existentes = len([name for name in os.listdir(caminho_palavra_dir) if name.endswith('.npy')])
                    caminho_arquivo_sequencia = os.path.join(caminho_palavra_dir, f"{num_sequencias_existentes + 1}.npy")
                    np.save(caminho_arquivo_sequencia, np.array(frames_desta_amostra)) # salva a sequência como .npy.
                    
                    print(f"Saved sample {num_sequencias_existentes + 1} for '{palavra_atual}' to {caminho_arquivo_sequencia}")
                    frames_desta_amostra = [] # reinicia a lista de frames.
                    coletando_amostra_atual = False # para a coleta desta amostra.
                    print("Press SPACE to record another sample, 'c' to change word, or 'q' to quit collection.")
            else:
                # instrução para iniciar a gravação.
                 cv2.putText(annotated_frame, "Press SPACE to record sample", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2) # Cyan.

            # painel de informações gerais.
            cv2.rectangle(annotated_frame, (0,0), (frame.shape[1], 40), (0,0,0), -1) # fundo preto.
            cv2.putText(annotated_frame, f"Word: {palavra_atual} | Press 'q' to quit, 'c' to change word", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2) # Branco.

            cv2.imshow('Word Sequence Collection', annotated_frame) # mostra o frame anotado.

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'): # sair do modo de coleta.
                break 
            if key == ord(' '): # iniciar/parar a coleta da amostra atual.
                if not coletando_amostra_atual:
                    coletando_amostra_atual = True
                    frames_desta_amostra = [] # reinicia os frames ao começar uma nova gravação.
                    print(f"Starting recording for sample of '{palavra_atual}'...")
            if key == ord('c'): # mudar para outra palavra.
                coletando_amostra_atual = False # para a coleta atual se estiver ocorrendo.
                frames_desta_amostra = []
                nova_palavra = escolher_palavra_para_coleta(palavras_para_escolha)
                if nova_palavra:
                    palavra_atual = nova_palavra
                    print(f"\nSwitched to collecting for word: '{palavra_atual}'")
                    print("Prepare and press SPACE to record a sample.")
                else:
                    print("No new word selected. Continuing with the current word or press 'q' to quit.")
        
    cap.release()
    cv2.destroyAllWindows()
    print("Exited collection mode.")

# carrega e pré-processa todos os dados de treino
# retorna x (dados de sequências), y (labels numéricas), lista ordenada de nomes de palavras únicas e mapeamento de id para nome da palavra
def carregar_dados_treino():
    # carrega todas as sequências de dados e rótulos (nomes das palavras) de ficheiros .npy no data_dir
    todas_sequencias_brutas = []
    todos_rotulos_texto_brutos = []
    # obtém lista de diretórios de palavras (cada subdiretório em data_dir representa uma palavra)
    palavras_detectadas = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    if not palavras_detectadas:
        print("Warning: No word data directories found in DATA_DIR. Cannot load training data.")
        return None, None, None, None

    for palavra in palavras_detectadas:
        caminho_dir_palavra = os.path.join(DATA_DIR, palavra)
        for nome_arquivo_sequencia in os.listdir(caminho_dir_palavra):
            if nome_arquivo_sequencia.endswith('.npy'):
                caminho_completo = os.path.join(caminho_dir_palavra, nome_arquivo_sequencia)
                try:
                    seq = np.load(caminho_completo)
                    # verifica se sequência tem o número correto de frames (sequence_length)
                    if seq.shape[0] == SEQUENCE_LENGTH:
                        todas_sequencias_brutas.append(seq)
                        todos_rotulos_texto_brutos.append(palavra)
                    else:
                        print(f"Warning: Sequence {caminho_completo} has length {seq.shape[0]}, expected {SEQUENCE_LENGTH}. Skipping.")
                except Exception as e:
                    print(f"Error loading or processing sequence file {caminho_completo}: {e}. Skipping.")
    
    if not todas_sequencias_brutas:
        print("Warning: No valid sequences loaded from data directories.")
        return None, None, None, None

    # filtra palavras sem número mínimo de amostras (min_samples_per_class_for_split)
    # importante para garantir que divisão estratificada (train_test_split) funcione corretamente
    contador_palavras = {}
    for rotulo in todos_rotulos_texto_brutos:
        contador_palavras[rotulo] = contador_palavras.get(rotulo, 0) + 1

    palavras_para_treino = []
    for palavra, contagem in contador_palavras.items():
        if contagem >= MIN_SAMPLES_PER_CLASS_FOR_SPLIT:
            palavras_para_treino.append(palavra)
        else:
            print(f"Warning: Word '{palavra}' has only {contagem} sample(s) (minimum required: {MIN_SAMPLES_PER_CLASS_FOR_SPLIT}). It will be excluded from the current training set.")

    if not palavras_para_treino:
        print(f"Error: No words have enough samples for training (at least {MIN_SAMPLES_PER_CLASS_FOR_SPLIT} samples per word required). Please collect more data.")
        return None, None, None, None

    # filtra sequências e rótulos para incluir apenas palavras com amostras suficientes
    sequencias_filtradas = []
    rotulos_texto_filtrados = []
    for i, rotulo in enumerate(todos_rotulos_texto_brutos):
        if rotulo in palavras_para_treino:
            sequencias_filtradas.append(todas_sequencias_brutas[i])
            rotulos_texto_filtrados.append(rotulo)
    
    if not sequencias_filtradas:
        # esta verificação pode ser redundante se a anterior (not palavras_para_treino) já cobrir
        print("Warning: No valid sequences remain after filtering by minimum sample count.")
        return None, None, None, None

    # cria mapeamentos entre nomes de palavras e ids numéricos
    lista_palavras_unicas_ordenada = sorted(list(set(rotulos_texto_filtrados)))
    mapa_palavra_para_id = {palavra: i for i, palavra in enumerate(lista_palavras_unicas_ordenada)}
    mapa_id_para_palavra = {i: palavra for palavra, i in mapa_palavra_para_id.items()}
        
    X = np.array(sequencias_filtradas)
    # converte rótulos de texto para numéricos (array 1d de inteiros)
    y_numeric = np.array([mapa_palavra_para_id[rotulo] for rotulo in rotulos_texto_filtrados])
    
    # print(f"Data loaded: X shape {X.shape}, y_numeric shape {y_numeric.shape}") # removido print informativo
    # print(f"Classes for training (ID to word map): {mapa_id_para_palavra}") # removido print informativo
    
    return X, y_numeric, lista_palavras_unicas_ordenada, mapa_id_para_palavra

# definição de número de features esperadas pelo modelo
# calculado com base no número de landmarks da mão (21*3) e corpo selecionados (9*3)
# mão: 21 landmarks * 3 coordenadas (x,y,z) = 63 features
# corpo: 9 landmarks selecionados * 3 coordenadas (x,y,z) = 27 features
# total = 63 + 27 = 90 features
NUM_FEATURES = (21 * 3) + (len([0, 11, 12, 13, 14, 15, 16, 23, 24]) * 3) 

# treina o modelo de reconhecimento de palavras
def treinar_modelo_palavras():
    # print("Iniciando treino do modelo de reconhecimento de palavras...") # removido print informativo
    
    # carrega dados de treino
    X, y_numeric, lista_palavras_unicas, mapa_id_para_palavra_json = carregar_dados_treino()
    
    if X is None or X.shape[0] == 0 or y_numeric is None or y_numeric.shape[0] == 0:
        print("Error: No training data loaded or data is insufficient. Aborting training.")
        return

    num_classes = len(lista_palavras_unicas)
    if num_classes < 2: # Alterado para exigir pelo menos 2 classes para treino com softmax/categorical_crossentropy
        print(f"Error: Number of classes ({num_classes}) is insufficient for training. At least 2 distinct classes with enough samples are required.")
        return

    # divide dados em conjuntos de treino e teste
    # estratificação (stratify=y_numeric) tenta manter proporção de cada classe em ambos os conjuntos
    test_size_ratio = 0.20 
    num_total_amostras = X.shape[0]
    num_amostras_teste_calculado = math.ceil(num_total_amostras * test_size_ratio)

    # verifica se conjunto de teste terá pelo menos uma amostra por classe (necessário para stratify)
    if num_amostras_teste_calculado < num_classes and num_classes > 1:
        print(f"Error: Calculated number of test samples ({num_amostras_teste_calculado}) is less than the number of classes ({num_classes}) for stratified split.")
        print(f"Total samples: {num_total_amostras}, Test ratio: {test_size_ratio*100}%.")
        print(f"To proceed, you need at least {math.ceil(num_classes / test_size_ratio)} total samples, or adjust test_size_ratio, or collect more data for minority classes.")
        return
        
    X_train, X_test, y_train_numeric, y_test_numeric = train_test_split(
        X, y_numeric, test_size=test_size_ratio, random_state=42, stratify=y_numeric if num_classes > 1 else None
    )

    # converte rótulos numéricos para formato one-hot categorical, necessário para 'categorical_crossentropy'
    y_train_categorical = tf.keras.utils.to_categorical(y_train_numeric, num_classes=num_classes)
    y_test_categorical = tf.keras.utils.to_categorical(y_test_numeric, num_classes=num_classes)

    # print(f"Shape of X_train: {X_train.shape}") # removido print informativo
    # print(f"Shape of y_train_categorical: {y_train_categorical.shape}") # removido print informativo
    
    # verifica se num_features corresponde à dimensão real dos dados carregados
    if X_train.shape[2] != NUM_FEATURES:
        print(f"Critical Error: NUM_FEATURES ({NUM_FEATURES}) does not match the feature dimension of loaded data ({X_train.shape[2]}).")
        print("This indicates a mismatch in keypoint extraction logic or NUM_FEATURES definition. Aborting training.")
        return 

    # define arquitetura do modelo lstm sequencial
    modelo = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
        tf.keras.layers.Dropout(0.3), # dropout para regularização, previne overfitting
        tf.keras.layers.LSTM(128, return_sequences=True, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64, return_sequences=False, activation='relu'), # return_sequences=false na última lstm antes da dense
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax') # camada de saída com ativação softmax para probabilidades de classe
    ])

    # compila o modelo, definindo otimizador, função de perda e métricas
    modelo.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    modelo.summary() # imprime resumo da arquitetura do modelo

    # callbacks para treino
    # earlystopping: para treino se perda na validação não melhorar após 'patience' épocas
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, min_delta=0.0005, restore_best_weights=True)
    # reducelronplateau: reduz taxa de aprendizagem se perda na validação estagnar (opcional)
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)

    print("Starting model training process...")
    history = modelo.fit(X_train, y_train_categorical, 
                         epochs=250, # número de épocas (pode ser alto devido a early stopping)
                         batch_size=32, 
                         validation_data=(X_test, y_test_categorical), 
                         callbacks=[early_stopping], # adicionar reduce_lr aqui se usado
                         verbose=1) # mostra progresso do treino (1 para barra, 2 para linha por época)
    
    # avalia modelo treinado no conjunto de teste
    test_loss, test_acc = modelo.evaluate(X_test, y_test_categorical, verbose=0)
    print(f"Model evaluation on test set: Accuracy: {test_acc*100:.2f}%, Loss: {test_loss:.4f}")
    
    # salva modelo treinado e mapeamento de classes (id para nome da palavra)
    modelo.save(MODEL_PATH)
    with open(CLASSES_PATH, 'w') as f:
        json.dump(mapa_id_para_palavra_json, f, indent=4) # salva mapa { "0": "palavraa", ... } com indentação
    print(f"Trained model saved to: {MODEL_PATH}")
    print(f"Class mapping (ID to word) saved to: {CLASSES_PATH}")

    # gera e imprime relatório de classificação detalhado
    # print("\nClassification Report (Detailed Metrics):") # removido print informativo
    y_pred_probabilities = modelo.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probabilities, axis=1) # converte probabilidades para classes (índices)
    
    # nomes das classes para relatório devem estar na ordem dos ids (0, 1, 2...)
    # assegura que variável correta 'mapa_id_para_palavra_json' é usada com chaves inteiras
    target_names_ordered = [mapa_id_para_palavra_json[i] for i in range(num_classes)]
    
    report = classification_report(y_test_numeric, y_pred_classes, target_names=target_names_ordered, zero_division=0)
    print("\nClassification Report:\n", report)

    # plota e salva histórico de treino (acurácia e perda)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    plt.tight_layout()
    
    plot_save_path = os.path.join(MODEL_DIR, "word_recognition_training_history.png")
    try:
        plt.savefig(plot_save_path)
        print(f"Training history plot saved to: {plot_save_path}")
    except Exception as e:
        print(f"Error saving training history plot: {e}")
    plt.close(fig) # fecha figura para libertar memória

# reconhece palavras gestuais usando modelo treinado
def reconhecer_palavras():
    # print("\n# Reconhecimento de Palavras Gestuais") # removido print informativo
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        print("Error: Trained model or class mapping file not found. Please train the model first (Option 2).")
        return

    mapa_id_para_palavra_loaded = None
    try:
        # carrega modelo keras treinado
        modelo = tf.keras.models.load_model(MODEL_PATH)
        # carrega mapeamento de id de classe para nome da palavra
        with open(CLASSES_PATH, 'r') as f:
            mapa_id_para_palavra_loaded = json.load(f) 
        print(f"Model '{MODEL_NAME}' and class mapping loaded. Model recognizes: {list(mapa_id_para_palavra_loaded.values())}")
    except Exception as e:
        print(f"Error loading the model or class mapping file: {e}")
        return # não pode continuar sem modelo

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    sequencia_atual_frames = [] # buffer para armazenar frames da sequência em tempo real
    palavra_prevista_display = "" # palavra a ser exibida na tela
    confianca_previsao_display = 0.0 # confiança da previsão a ser exibida
    limiar_confianca = 0.65 # limiar de confiança para considerar previsão válida
    
    frames_desde_ultima_previsao = 0
    INTERVALO_PREVISAO = 5 # faz previsão a cada x frames para otimizar (evita sobrecarga)

    # variáveis para cálculo de métricas de desempenho em tempo real
    start_time_recognition_loop = time.time()
    frame_counter_total = 0
    tempo_total_extracao_pontos_ms = 0
    tempo_total_previsao_modelo_ms = 0
    num_previsoes_efetuadas = 0

    # bloco 'with' para gestão de recursos mediapipe
    with mp_hands.Hands(max_num_hands=MAX_NUM_HANDS, min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as hands_model, \
         mp_pose.Pose(min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as pose_model:
        
        while cap.isOpened():
            frame_counter_total +=1
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame from camera.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False # otimização
            results_hands = hands_model.process(frame_rgb)
            results_pose = pose_model.process(frame_rgb)
            frame_rgb.flags.writeable = True
            annotated_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # extrai pontos chave do frame atual e mede tempo de extração
            ts_extracao_inicio = time.perf_counter()
            pontos_chave_frame_atual, _ = extrair_pontos_chave(results_hands, results_pose)
            ts_extracao_fim = time.perf_counter()
            tempo_total_extracao_pontos_ms += (ts_extracao_fim - ts_extracao_inicio) * 1000
            
            sequencia_atual_frames.append(pontos_chave_frame_atual)
            
            # mantém buffer da sequência no tamanho correto (janela deslizante)
            if len(sequencia_atual_frames) > SEQUENCE_LENGTH:
                sequencia_atual_frames = sequencia_atual_frames[-SEQUENCE_LENGTH:]
            
            frames_desde_ultima_previsao += 1

            # faz previsão se buffer cheio e intervalo de frames para previsão atingido
            if len(sequencia_atual_frames) == SEQUENCE_LENGTH and frames_desde_ultima_previsao >= INTERVALO_PREVISAO:
                frames_desde_ultima_previsao = 0
                # prepara sequência para formato esperado pelo modelo (batch_size=1)
                sequencia_para_previsao = np.expand_dims(np.array(sequencia_atual_frames), axis=0)
                
                ts_previsao_inicio = time.perf_counter()
                prediction_probabilities = modelo.predict(sequencia_para_previsao, verbose=0)[0]
                ts_previsao_fim = time.perf_counter()
                tempo_total_previsao_modelo_ms += (ts_previsao_fim - ts_previsao_inicio) * 1000
                num_previsoes_efetuadas += 1

                id_previsto = np.argmax(prediction_probabilities)
                confianca_previsao_display = prediction_probabilities[id_previsto]
                
                # limiar de confiança para considerar previsão válida
                if confianca_previsao_display > limiar_confianca:
                    # usa str(id_previsto) porque chaves no mapa json são strings
                    palavra_prevista_display = mapa_id_para_palavra_loaded.get(str(id_previsto), "Error: ID?")
                else:
                    palavra_prevista_display = "-" # indica incerteza ou nenhum gesto reconhecido
            
            # desenha landmarks das mãos
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            
            # desenha landmarks do corpo
            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # painel de informações com previsão e métricas de desempenho
            cv2.rectangle(annotated_frame, (0,0), (annotated_frame.shape[1], 110), (0,0,0), -1)
            cor_texto_previsao = (0,255,0) if confianca_previsao_display > limiar_confianca and palavra_prevista_display != "-" else (0,165,255) # verde para confiante, laranja para incerto/nenhum
            cv2.putText(annotated_frame, f"Word: {palavra_prevista_display.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, cor_texto_previsao, 2)
            cv2.putText(annotated_frame, f"Confidence: {confianca_previsao_display*100:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_texto_previsao, 2)
            cv2.putText(annotated_frame, f"Buffer: {len(sequencia_atual_frames)}/{SEQUENCE_LENGTH}", (annotated_frame.shape[1]-280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
            
            # cálculo e exibição de métricas de desempenho (fps, tempo de extração, tempo de previsão)
            if frame_counter_total > 0:
                tempo_decorrido_total_loop = time.time() - start_time_recognition_loop
                avg_fps = frame_counter_total / tempo_decorrido_total_loop if tempo_decorrido_total_loop > 0 else 0
                avg_tempo_extracao_ms = tempo_total_extracao_pontos_ms / frame_counter_total if frame_counter_total > 0 else 0
                avg_tempo_previsao_ms = tempo_total_previsao_modelo_ms / num_previsoes_efetuadas if num_previsoes_efetuadas > 0 else 0

            cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2) # amarelo
            cv2.putText(annotated_frame, f"KeyExtract: {avg_tempo_extracao_ms:.1f}ms", (annotated_frame.shape[1]-280, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)
            cv2.putText(annotated_frame, f"ModelPred: {avg_tempo_previsao_ms:.1f}ms", (annotated_frame.shape[1]-280, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)

            cv2.putText(annotated_frame, "[Q] Quit Recognition", (10, annotated_frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Word Gesture Recognition", annotated_frame)
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Exited recognition mode.")

# remove todos os dados de uma palavra específica
def remover_palavra_dados():
    # print("\n# Remover Palavra e Seus Dados") # removido print informativo
    
    palavras_existentes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if not palavras_existentes:
        print("No word data found to remove.")
        return

    print("\nAvailable words for removal:")
    palavras_ordenadas_para_remocao = sorted(palavras_existentes)
    for idx, palavra in enumerate(palavras_ordenadas_para_remocao):
        print(f"{idx + 1} - {palavra}")
    
    try:
        escolha_idx_str = input("Choose the number of the word to remove (or 'c' to cancel): ").strip().lower()
        if escolha_idx_str == 'c':
            print("Removal cancelled.")
            return

        idx_para_remover = int(escolha_idx_str) - 1
        # valida índice escolhido
        if 0 <= idx_para_remover < len(palavras_ordenadas_para_remocao):
            palavra_para_remover = palavras_ordenadas_para_remocao[idx_para_remover]
            confirmacao = input(f"Are you sure you want to PERMANENTLY remove the word '{palavra_para_remover}' and all its sequence data? (yes/no): ").strip().lower()
            if confirmacao == 'yes':
                caminho_palavra_dir_remover = os.path.join(DATA_DIR, palavra_para_remover)
                try:
                    shutil.rmtree(caminho_palavra_dir_remover) # remove diretório da palavra e todo o seu conteúdo
                    print(f"Word '{palavra_para_remover}' and all its data have been successfully removed.")
                    
                    if os.path.exists(MODEL_PATH) or os.path.exists(CLASSES_PATH):
                        print("Warning: A trained model currently exists. It is highly recommended to retrain the model (Option 2)")
                        print("as it might still be configured to recognize the removed word, potentially leading to incorrect behavior.")

                except Exception as e:
                    print(f"Error removing the directory for word '{palavra_para_remover}': {e}")
            else:
                print("Removal cancelled.")
        else:
            print("Invalid selection number.")
    except ValueError:
        print("Invalid input. Please enter a number corresponding to the word or 'c' to cancel.")
    # except IndexError: # esta exceção menos provável se validação de idx_para_remover correta
    #     print("Number out of range. Please choose a number from the list.")

# loop principal da aplicação
def main_loop():
    while True:
        clear() # limpa terminal antes de mostrar menu
        opcao = menu_principal() # mostra menu e obtém escolha do utilizador
        
        if opcao == '1':
            coletar_sequencias_palavras()
        elif opcao == '2':
            treinar_modelo_palavras()
        elif opcao == '3':
            reconhecer_palavras()
        elif opcao == '4':
            remover_palavra_dados()
        elif opcao == 'q':
            # print("Exiting program.") # removido print informativo
            break
        else:
            print("Invalid option. Please try again.")
        input("\nPress Enter to continue...") # pausa para utilizador ver output antes de limpar ecrã

if __name__ == "__main__":
    # ponto de entrada do script
    main_loop() 