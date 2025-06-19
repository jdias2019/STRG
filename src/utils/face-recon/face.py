import threading
import cv2
import os
import json
import numpy as np
import pickle
from deepface import DeepFace
from datetime import datetime
import time
import platform
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class FaceRecognitionSystem:
    def __init__(self):
        # configurar câmara baseado no sistema operativo
        if platform.system() == "Windows":
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            # Linux/MacOS - tentar diferentes backends
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                # tentar com V4L2 no Linux
                self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        # verificar se a câmara foi aberta com sucesso
        if not self.cap.isOpened():
            raise RuntimeError("no webcam")
        
        # configurar resolução
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # testar captura de frame
        ret, test_frame = self.cap.read()
        if not ret:
            raise RuntimeError("no frame")
        
        print(f"✓ Câmara inicializada com sucesso - Resolução: {test_frame.shape[1]}x{test_frame.shape[0]}")
        
        # obter diretório atual do script
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # diretório para armazenar as faces treinadas (dentro do face-recon)
        self.faces_dir = os.path.join(self.script_dir, "trained_faces")
        self.database_file = os.path.join(self.script_dir, "face_database.json")
        self.model_file = os.path.join(self.script_dir, "face_model.pkl")
        self.embeddings_file = os.path.join(self.script_dir, "face_embeddings.pkl")
        
        # criar diretório se não existir
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)
            print(f"✓ Diretório criado: {self.faces_dir}")
        
        # carregar base de dados e modelo existentes
        self.face_database = self.load_database()
        self.face_embeddings = []
        self.face_labels = []
        self.label_encoder = LabelEncoder()
        self.classifier = None
        self.is_model_trained = False
        
        # carregar modelo treinado se existir
        self.load_trained_model()
        
        # variáveis para reconhecimento em tempo real
        self.current_matches = {}
        self.recognition_lock = threading.Lock()
        self.counter = 0
        
        # configurações
        self.confidence_threshold = 0.7  # limiar de confiança para classificação
        self.recognition_interval = 15  # verificar a cada 15 frames para melhor responsividade
        self.model_name = 'Facenet512'  # modelo mais preciso para embeddings
        
    def load_database(self):
        """carrega a base de dados de faces treinadas"""
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_database(self):
        """guarda a base de dados de faces"""
        with open(self.database_file, 'w', encoding='utf-8') as f:
            json.dump(self.face_database, f, indent=2, ensure_ascii=False)
    
    def load_trained_model(self):
        """carrega o modelo treinado se existir"""
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.embeddings_file):
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    self.classifier = model_data['classifier']
                    self.label_encoder = model_data['label_encoder']
                
                with open(self.embeddings_file, 'rb') as f:
                    embeddings_data = pickle.load(f)
                    self.face_embeddings = embeddings_data['embeddings']
                    self.face_labels = embeddings_data['labels']
                
                self.is_model_trained = True
                print(f"✓ Modelo carregado com {len(self.face_embeddings)} amostras")
                print(f"✓ Classes treinadas: {list(self.label_encoder.classes_)}")
        except Exception as e:
            print(f"Aviso: Erro ao carregar modelo: {e}")
            self.is_model_trained = False
    
    def save_trained_model(self):
        """guarda o modelo treinado"""
        try:
            model_data = {
                'classifier': self.classifier,
                'label_encoder': self.label_encoder
            }
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            embeddings_data = {
                'embeddings': self.face_embeddings,
                'labels': self.face_labels
            }
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(embeddings_data, f)
            
            print("✓ Modelo guardado com sucesso")
        except Exception as e:
            print(f"Erro ao guardar modelo: {e}")
    
    def extract_face_embedding(self, image):
        """extrai embedding de uma face"""
        try:
            # usar DeepFace para extrair embedding
            embedding = DeepFace.represent(image, model_name=self.model_name, enforce_detection=False)
            return np.array(embedding[0]['embedding'])
        except Exception as e:
            print(f"Erro ao extrair embedding: {e}")
            return None
    
    def train_face(self, name, num_samples=15):
        """treina uma nova face capturando múltiplas amostras e criando embeddings"""
        print(f"A treinar face para: {name}")
        print("Prima 'c' para capturar uma amostra, 'q' para terminar")
        print("Se a janela não abrir, verifique se tem permissões para usar a câmara")
        print("Nota: Varie a posição da face (frente, lado, diferentes expressões)")
        
        person_dir = os.path.join(self.faces_dir, name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        samples_captured = 0
        valid_embeddings = []
        window_name = 'Treinar Face'
        
        # criar janela com propriedades específicas
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        while samples_captured < num_samples:
            ret, frame = self.cap.read()
            if not ret:
                print("Erro: Não foi possível capturar frame da câmara")
                time.sleep(0.1)
                continue
            
            # mostrar instruções no frame
            cv2.putText(frame, f"Treinar: {name}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Amostras: {samples_captured}/{num_samples}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Prima 'c' para capturar, 'q' para sair", (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Varie a posicao da face!", (20, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            try:
                cv2.imshow(window_name, frame)
            except cv2.error as e:
                print(f"Erro ao mostrar frame: {e}")
                print("Tentando continuar...")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                print(f"Processando amostra {samples_captured + 1}...")
                
                # extrair embedding da face
                embedding = self.extract_face_embedding(frame)
                
                if embedding is not None:
                    # guardar imagem
                    filename = f"{name}_{samples_captured}_{int(time.time())}.jpg"
                    filepath = os.path.join(person_dir, filename)
                    
                    if cv2.imwrite(filepath, frame):
                        # adicionar à base de dados
                        if name not in self.face_database:
                            self.face_database[name] = []
                        
                        self.face_database[name].append({
                            'filename': filename,
                            'filepath': filepath,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # adicionar embedding aos dados de treino
                        valid_embeddings.append(embedding)
                        samples_captured += 1
                        print(f"✓ Amostra {samples_captured} capturada e processada para {name}")
                    else:
                        print("Erro ao guardar imagem")
                else:
                    print("Erro: Face não detectada ou embedding inválido. Tente novamente.")
                
            elif key == ord('q'):
                break
        
        cv2.destroyWindow(window_name)
        
        # adicionar embeddings aos dados de treino
        if valid_embeddings:
            self.face_embeddings.extend(valid_embeddings)
            self.face_labels.extend([name] * len(valid_embeddings))
            
            self.save_database()
            print(f"✓ Treino concluído para {name} com {len(valid_embeddings)} embeddings válidos")
            
            # retreinar o modelo
            self.train_classifier()
        else:
            print("❌ Nenhuma amostra válida capturada!")
    
    def train_classifier(self):
        """treina o classificador com os embeddings coletados"""
        if len(self.face_embeddings) < 2:
            print("Aviso: Precisa de pelo menos 2 amostras para treinar o classificador")
            return False
        
        if len(set(self.face_labels)) < 2:
            print("Aviso: Precisa de pelo menos 2 pessoas diferentes para treinar o classificador")
            return False
        
        try:
            print("A treinar classificador...")
            
            # converter para arrays numpy
            X = np.array(self.face_embeddings)
            y = np.array(self.face_labels)
            
            # codificar labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # dividir dados para treino e teste
            if len(X) > 4:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )
            else:
                X_train, y_train = X, y_encoded
                X_test, y_test = X, y_encoded
            
            # treinar classificador SVM
            self.classifier = SVC(
                kernel='rbf',
                probability=True,
                C=1.0,
                gamma='scale',
                random_state=42
            )
            
            self.classifier.fit(X_train, y_train)
            
            # avaliar performance
            y_pred = self.classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_model_trained = True
            
            print(f"✓ Classificador treinado com sucesso!")
            print(f"✓ Precisão: {accuracy:.2%}")
            print(f"✓ Amostras de treino: {len(X_train)}")
            print(f"✓ Classes: {list(self.label_encoder.classes_)}")
            
            # guardar modelo
            self.save_trained_model()
            
            return True
            
        except Exception as e:
            print(f"Erro ao treinar classificador: {e}")
            self.is_model_trained = False
            return False
    
    def recognize_faces(self, frame):
        """reconhece faces no frame atual usando o classificador treinado"""
        if not self.is_model_trained:
            return {}
        
        matches = {}
        
        try:
            # extrair embedding da face no frame atual
            embedding = self.extract_face_embedding(frame)
            
            if embedding is None:
                return matches
            
            # usar o classificador para prever a pessoa
            embedding_reshaped = embedding.reshape(1, -1)
            
            # obter probabilidades para todas as classes
            probabilities = self.classifier.predict_proba(embedding_reshaped)[0]
            predicted_class = self.classifier.predict(embedding_reshaped)[0]
            
            # obter o nome da pessoa prevista
            predicted_name = self.label_encoder.inverse_transform([predicted_class])[0]
            confidence = probabilities[predicted_class]
            
            # apenas retornar se a confiança for suficiente
            if confidence >= self.confidence_threshold:
                matches[predicted_name] = confidence
                
                # mostrar top 3 previsões para debug (apenas se verbose)
                if hasattr(self, 'verbose') and self.verbose:
                    top_indices = np.argsort(probabilities)[::-1][:3]
                    print(f"Top previsões: ", end="")
                    for i, idx in enumerate(top_indices):
                        name = self.label_encoder.inverse_transform([idx])[0]
                        prob = probabilities[idx]
                        print(f"{name}({prob:.2f}) ", end="")
                    print()
            
        except Exception as e:
            print(f"Erro no reconhecimento: {e}")
        
        return matches
    
    def update_recognition(self, frame):
        """atualiza o reconhecimento em thread separada"""
        matches = self.recognize_faces(frame)
        
        with self.recognition_lock:
            self.current_matches = matches
    
    def draw_results(self, frame):
        """desenha os resultados no frame"""
        with self.recognition_lock:
            # mostrar status do modelo
            if self.is_model_trained:
                status_text = f"Modelo: Treinado ({len(self.face_embeddings)} amostras)"
                cv2.putText(frame, status_text, (20, 450), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                cv2.putText(frame, "Modelo: Nao treinado", (20, 450), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # mostrar reconhecimentos
            if self.current_matches:
                y_offset = 50
                for person, confidence in self.current_matches.items():
                    text = f"{person}: {confidence:.2%}"
                    color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
                    cv2.putText(frame, text, (20, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    y_offset += 40
            else:
                if self.is_model_trained:
                    cv2.putText(frame, "Nenhuma face reconhecida", (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Treine o modelo primeiro!", (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    def run_recognition(self):
        """executa o reconhecimento em tempo real"""
        print("Sistema de reconhecimento iniciado")
        print("Controlos:")
        print("- 't': treinar nova face")
        print("- 'r': modo reconhecimento")
        print("- 'q': sair")
        print("Se a janela não abrir, pressione Ctrl+C para sair")
        
        mode = "recognition"
        window_name = 'Sistema de Reconhecimento Facial'
        
        # criar janela
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Erro: Não foi possível capturar frame")
                    time.sleep(0.1)
                    continue
                
                # mostrar modo atual
                cv2.putText(frame, f"Modo: {mode}", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if mode == "recognition":
                    # reconhecimento em tempo real
                    if self.counter % self.recognition_interval == 0:
                        try:
                            threading.Thread(target=self.update_recognition, 
                                           args=(frame.copy(),), daemon=True).start()
                        except Exception as e:
                            print(f"Erro no reconhecimento: {e}")
                    
                    self.draw_results(frame)
                    self.counter += 1
                
                try:
                    cv2.imshow(window_name, frame)
                except cv2.error as e:
                    print(f"Erro ao mostrar frame: {e}")
                    break
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    # modo treino
                    cv2.destroyWindow(window_name)
                    name = input("Introduza o nome da pessoa para treinar: ").strip()
                    if name:
                        self.train_face(name)
                        mode = "recognition"
                    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                elif key == ord('r'):
                    mode = "recognition"
        
        except KeyboardInterrupt:
            print("\nInterrompido pelo utilizador")
        finally:
            cv2.destroyWindow(window_name)
            self.cleanup()
    
    def list_trained_faces(self):
        """lista todas as faces treinadas"""
        if not self.face_database:
            print("Nenhuma face treinada encontrada")
            return
        
        print("Faces treinadas:")
        for name, samples in self.face_database.items():
            print(f"- {name}: {len(samples)} amostras")
    
    def retrain_from_database(self):
        """retreina o modelo usando todas as imagens na base de dados"""
        if not self.face_database:
            print("Nenhuma face na base de dados para retreinar")
            return False
        
        print("A retreinar modelo com todas as faces da base de dados...")
        
        # limpar embeddings atuais
        self.face_embeddings = []
        self.face_labels = []
        
        total_processed = 0
        
        for person_name, samples in self.face_database.items():
            print(f"Processando {person_name}...")
            person_embeddings = 0
            
            for sample in samples:
                filepath = sample['filepath']
                if os.path.exists(filepath):
                    try:
                        # carregar imagem
                        image = cv2.imread(filepath)
                        if image is not None:
                            # extrair embedding
                            embedding = self.extract_face_embedding(image)
                            if embedding is not None:
                                self.face_embeddings.append(embedding)
                                self.face_labels.append(person_name)
                                person_embeddings += 1
                                total_processed += 1
                    except Exception as e:
                        print(f"Erro ao processar {filepath}: {e}")
            
            print(f"  {person_embeddings} embeddings extraídos para {person_name}")
        
        print(f"Total de {total_processed} embeddings processados")
        
        if total_processed > 0:
            # treinar classificador
            success = self.train_classifier()
            if success:
                print("✓ Modelo retreinado com sucesso!")
            return success
        else:
            print("❌ Nenhum embedding válido encontrado")
            return False
    
    def delete_person(self, name):
        """remove uma pessoa da base de dados"""
        if name in self.face_database:
            # remover ficheiros
            person_dir = os.path.join(self.faces_dir, name)
            if os.path.exists(person_dir):
                import shutil
                shutil.rmtree(person_dir)
            
            # remover embeddings da pessoa
            indices_to_remove = [i for i, label in enumerate(self.face_labels) if label == name]
            for idx in reversed(indices_to_remove):  # remover de trás para frente
                del self.face_embeddings[idx]
                del self.face_labels[idx]
            
            # remover da base de dados
            del self.face_database[name]
            self.save_database()
            
            # retreinar modelo se ainda houver dados
            if self.face_embeddings:
                self.train_classifier()
            else:
                self.is_model_trained = False
            
            print(f"Pessoa {name} removida da base de dados")
        else:
            print(f"Pessoa {name} não encontrada na base de dados")
    
    def cleanup(self):
        """limpa recursos"""
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    """função principal com menu interativo"""
    system = FaceRecognitionSystem()
    
    while True:
        print("\n=== Sistema de Reconhecimento Facial Avançado ===")
        print("1. Iniciar reconhecimento em tempo real")
        print("2. Treinar nova face")
        print("3. Listar faces treinadas")
        print("4. Retreinar modelo completo")
        print("5. Remover pessoa")
        print("6. Sair")
        
        if system.is_model_trained:
            print(f"Status: Modelo treinado com {len(system.face_embeddings)} amostras")
        else:
            print("Status: Modelo não treinado")
        
        choice = input("Escolha uma opção: ").strip()
        
        if choice == '1':
            if system.is_model_trained:
                system.run_recognition()
            else:
                print("❌ Precisa treinar pelo menos 2 pessoas diferentes primeiro!")
        elif choice == '2':
            name = input("Nome da pessoa: ").strip()
            if name:
                num_samples = input("Número de amostras (padrão 15): ").strip()
                try:
                    num_samples = int(num_samples) if num_samples else 15
                except:
                    num_samples = 15
                system.train_face(name, num_samples)
        elif choice == '3':
            system.list_trained_faces()
        elif choice == '4':
            system.retrain_from_database()
        elif choice == '5':
            name = input("Nome da pessoa a remover: ").strip()
            if name:
                system.delete_person(name)
        elif choice == '6':
            break
        else:
            print("Opção inválida")
    
    system.cleanup()


if __name__ == "__main__":
    main() 