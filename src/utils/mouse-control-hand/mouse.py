# controlo de cursor por gestos - versão otimizada
import cv2
import numpy as np
import time
from pynput.mouse import Button, Controller
import tkinter as tk
import sys
import os
import gc

# configuração de caminhos
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from utils import HandTrackingModule as htm
    from utils.shared_utils import CameraManager, add_performance_overlay, cleanup_cv_resources
except ImportError:
    # fallback para estrutura antiga
    import HandTrackingModule as htm


class OptimizedMouseController:
    # controlador de cursor otimizado
    
    def __init__(self):
        # configuração
        self.config = {
            'frame_reduction': 80,
            'smoothing': 5,
            'click_threshold': 40,
            'fps_limit': 30,
            'reset_interval': 300  # reiniciar detector a cada 300 frames
        }
        
        # estado interno
        self.prev_loc = np.array([0, 0])
        self.curr_loc = np.array([0, 0])
        self.last_click_time = 0
        self.click_cooldown = 0.3
        self.frame_counter = 0
        self.debug_mode = True  # modo de debug para visualizar valores
        
        # contadores internos
        self.click_count = 0
        self.total_cursor_distance = 0.0
        self.hands_detected_count = 0
        
        # inicialização de componentes
        self._setup_screen()
        self._setup_camera()
        self._setup_detector()
        self._setup_mouse()
        
    def _setup_screen(self):
        # obtém dimensões do ecrã
        try:
            root = tk.Tk()
            root.withdraw()
            self.screen_size = np.array([root.winfo_screenwidth(), root.winfo_screenheight()])
            root.destroy()
            print(f"Tamanho do ecrã: {self.screen_size}")
        except Exception as e:
            print(f"Erro ao obter tamanho do ecrã: {e}")
            self.screen_size = np.array([1920, 1080])  # fallback
    
    def _setup_camera(self):
        # inicializa gestão da câmara
        self.camera = CameraManager(640, 480, self.config['fps_limit'])
        if not self.camera.initialize():
            raise RuntimeError("Falha ao inicializar câmara")
        print("Câmara inicializada com sucesso")
    
    def _setup_detector(self):
        # configura detetor de mãos
        self.detector = htm.handDetector(maxHands=1, detectionCon=0.7, trackCon=0.7)
        print("Detector de mãos configurado")
    
    def _setup_mouse(self):
        # inicializa controlador do cursor
        self.mouse = Controller()
        # Verificar se o controlador do rato está a funcionar
        initial_pos = self.mouse.position
        print(f"Posição inicial do cursor: {initial_pos}")
    
    def _get_finger_position(self, landmarks):
        # extrai posição do dedo indicador
        if len(landmarks) > 8:
            return np.array(landmarks[8][1:3])  # apenas x, y
        return None
    
    def _map_to_screen(self, hand_pos, frame_size):
        # mapeia posição da mão para coordenadas do ecrã
        frame_w, frame_h = frame_size
        margin = self.config['frame_reduction']
        
        # área útil da imagem
        active_area = np.array([frame_w - 2*margin, frame_h - 2*margin])
        
        # Garantir que não dividimos por zero
        if active_area[0] == 0 or active_area[1] == 0:
            return self.prev_loc
            
        # Garantir que hand_pos está dentro dos limites
        hand_pos_clipped = np.clip(hand_pos, margin, np.array([frame_w - margin, frame_h - margin]))
        
        # Calcular posição relativa
        relative_pos = (hand_pos_clipped - margin) / active_area
        
        # limitar e mapear para ecrã
        relative_pos = np.clip(relative_pos, 0, 1)
        screen_pos = relative_pos * self.screen_size
        
        if self.debug_mode:
            print(f"Hand: {hand_pos}, Relative: {relative_pos}, Screen: {screen_pos}")
            
        return screen_pos
    
    def _smooth_movement(self, target_pos):
        # aplica suavização ao movimento
        smoothing = self.config['smoothing']
        
        # Verificar se target_pos é válido
        if np.isnan(target_pos).any() or np.isinf(target_pos).any():
            return self.curr_loc
            
        self.curr_loc = self.prev_loc + (target_pos - self.prev_loc) / smoothing
        self.prev_loc = self.curr_loc.copy()
        return self.curr_loc
    
    def _handle_gestures(self, landmarks, frame):
        # processa gestos da mão
        fingers = self.detector.fingersUp()
        if not fingers or len(fingers) < 5:
            if self.debug_mode:
                print(f"Dedos não detectados corretamente: {fingers}")
            return frame
        
        finger_pos = self._get_finger_position(landmarks)
        if finger_pos is None:
            return frame
        
        frame_h, frame_w = frame.shape[:2]
        
        # área de controlo visual
        margin = self.config['frame_reduction']
        cv2.rectangle(frame, (margin, margin), (frame_w - margin, frame_h - margin), (255, 0, 255), 2)
        
        # movimento: apenas indicador
        if fingers[1] == 1 and fingers[2] == 0:
            screen_pos = self._map_to_screen(finger_pos, (frame_w, frame_h))
            smooth_pos = self._smooth_movement(screen_pos)
            
            # Garantir que a posição é válida
            if not np.isnan(smooth_pos).any() and not np.isinf(smooth_pos).any():
                try:
                    # calcula distância percorrida
                    if hasattr(self, 'last_cursor_pos'):
                        distance = np.linalg.norm(smooth_pos - self.last_cursor_pos)
                        self.total_cursor_distance += distance
                    self.last_cursor_pos = smooth_pos.copy()
                    
                    self.mouse.position = tuple(smooth_pos.astype(int))
                    if self.debug_mode:
                        print(f"Movendo cursor para: {tuple(smooth_pos.astype(int))}")
                except Exception as e:
                    print(f"Erro ao mover cursor: {e}")
                    
            cv2.circle(frame, tuple(finger_pos.astype(int)), 10, (255, 0, 255), cv2.FILLED)
        
        # clique: indicador + médio próximos
        elif fingers[1] == 1 and fingers[2] == 1:
            current_time = time.time()
            if current_time - self.last_click_time > self.click_cooldown:
                distance, frame, _ = self.detector.findDistance(8, 12, frame)
                
                if distance < self.config['click_threshold']:
                    try:
                        self.mouse.click(Button.left)
                        self.click_count += 1
                        if self.debug_mode:
                            print("Clique detectado!")
                        self.last_click_time = current_time
                        cv2.circle(frame, tuple(finger_pos.astype(int)), 15, (0, 255, 0), cv2.FILLED)
                    except Exception as e:
                        print(f"Erro ao clicar: {e}")
        
        return frame
    
    def _reset_detector_if_needed(self):
        # reinicia o detector periodicamente para evitar vazamentos de memória
        self.frame_counter += 1
        if self.frame_counter >= self.config['reset_interval']:
            self.detector = None
            gc.collect()  # força coleta de lixo
            self._setup_detector()
            self.frame_counter = 0
            print("Detector reiniciado para manter performance")
    
    def run(self):
        # executa controlo principal
        fps_tracker = time.time()
        frame_count = 0
        fps = 0
        
        print("Controlo de cursor iniciado. Prima 'q' para sair.")
        
        try:
            while True:
                start_time = time.time()
                success, frame = self.camera.read_frame()
                if not success:
                    print("Falha ao ler frame da câmara")
                    time.sleep(0.1)
                    continue
                
                # espelhar para intuição natural
                frame = cv2.flip(frame, 1)
                
                # deteção de mãos
                frame = self.detector.findHands(frame)
                landmarks, _ = self.detector.findPosition(frame, draw=False)
                

                
                if landmarks:
                    self.hands_detected_count += 1
                    frame = self._handle_gestures(landmarks, frame)
                else:
                    if self.debug_mode and frame_count % 30 == 0:  # Mostrar apenas a cada 30 frames
                        print("Nenhuma mão detectada")
                
                # reiniciar detector periodicamente
                self._reset_detector_if_needed()
                
                # cálculo correto de FPS
                frame_count += 1
                current_time = time.time()
                elapsed = current_time - fps_tracker
                
                # atualiza FPS a cada segundo
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    fps_tracker = current_time
                    frame_count = 0
                    if self.debug_mode:
                        print(f"FPS: {fps:.1f}")
                    

                
                frame = add_performance_overlay(frame, fps, "Cursor Control")
                
                # limitar taxa de frames se processamento for muito rápido
                process_time = time.time() - start_time
                wait_time = max(1, int((1.0/self.config['fps_limit'] - process_time) * 1000))
                
                cv2.imshow("Controlo de Cursor", frame)
                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Erro inesperado: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        # limpeza de recursos
        self.camera.release()
        cleanup_cv_resources()
        print("Recursos libertados.")


def main():
    # ponto de entrada principal
    try:
        controller = OptimizedMouseController()
        controller.run()
    except Exception as e:
        print(f"Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 