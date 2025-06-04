import cv2
import numpy as np

# callback para trackbar de limiar
# chamada sempre que valor do trackbar muda
# x: valor atual do trackbar
# não faz nada, mas necessária para api opencv
def ao_mudar_limiar(x: int) -> None:
    # print(f"Threshold changed to: {x}") # para depuração, se necessário
    pass

# converte imagem bgr para binária usando valor de limiar
# img: imagem de entrada bgr (colorida)
# limiar_value: valor inteiro (0-255) para limiarização. pixels mais escuros
# que o limiar ficam pretos, mais claros ficam brancos
# retorna imagem binária (preto e branco)
def binarizar_com_limiar_dinamico(img: np.ndarray, limiar_value: int) -> np.ndarray:
    # converte imagem colorida para escala de cinza
    # limiarização geralmente aplicada em imagens de um canal
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # aplica limiarização binária
    # cv2.thresh_binary: se pixel_value > limiar_value, é maxval (255), senão 0
    # retorna limiar usado (útil com métodos automáticos como otsu) e imagem limiarizada
    _, binary_img = cv2.threshold(gray_img, limiar_value, 255, cv2.THRESH_BINARY)
    return binary_img

# encontra contornos em imagem binária e desenha-os em cópia da original
# img_bin: imagem binária para procurar contornos
# original_para_desenho: imagem original (colorida/cinza) para desenhar contornos
# usar original para desenho preserva cores e detalhes do fundo
# retorna nova imagem com contornos desenhados
def mostrar_contornos(img_bin: np.ndarray, original_para_desenho: np.ndarray) -> np.ndarray:
    # encontra contornos na imagem binária
    # cv2.retr_external: recupera apenas contornos externos mais extremos
    # cv2.chain_approx_simple: comprime segmentos horizontais, verticais e diagonais,
    # deixando apenas pontos finais. poupa memória
    contornos, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # cria cópia da imagem original para não modificar fonte diretamente
    contorno_img_desenhada = original_para_desenho.copy()
    
    # desenha todos os contornos encontrados na imagem
    # -1: desenha todos os contornos
    # (0, 255, 0): cor verde para contornos
    # 2: espessura da linha do contorno
    cv2.drawContours(contorno_img_desenhada, contornos, -1, (0, 255, 0), 2)
    return contorno_img_desenhada

# aplica operações morfológicas (dilatação, erosão) a imagem binária
# img_bin: imagem binária de entrada
# retorna imagem dilatada e imagem erodida
def morfologia(img_bin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # define elemento estruturante (kernel) para operações
    # kernel 5x5 de 'uns': vizinhança de cada pixel é quadrado 5x5
    kernel = np.ones((5, 5), np.uint8)
    
    # dilatação: expande regiões brancas (primeiro plano) da imagem
    # útil para preencher pequenos buracos ou juntar objetos próximos
    dilatada_img = cv2.dilate(img_bin, kernel, iterations=1)
    
    # erosão: encolhe regiões brancas da imagem
    # útil para remover pequenos ruídos brancos ou separar objetos ligados
    erodida_img = cv2.erode(img_bin, kernel, iterations=1)
    return dilatada_img, erodida_img

# processa frame (ou imagem estática) e exibe diferentes vistas
# frame_para_processar: frame atual webcam ou imagem carregada para aplicar binarização/morfologia
# original_frame_para_contornos: frame original usado como fundo para desenhar contornos
# garante que contornos são sobrepostos na imagem original colorida, não na binária
def processar_e_mostrar_frame(frame_para_processar: np.ndarray, original_frame_para_contornos: np.ndarray) -> None:
    # obtém valor atual do limiar do trackbar na janela 'controlos'
    limiar_atual = cv2.getTrackbarPos('Limiar', 'Controlos')
    
    # aplica transformações à imagem
    img_binaria = binarizar_com_limiar_dinamico(frame_para_processar, limiar_atual)
    img_contornos = mostrar_contornos(img_binaria, original_frame_para_contornos)
    img_dilatada, img_erodida = morfologia(img_binaria)
    
    # exibe imagens resultantes em janelas separadas
    cv2.imshow("Binary", img_binaria)
    cv2.imshow("Contours", img_contornos)
    cv2.imshow("Dilated", img_dilatada)
    cv2.imshow("Eroded", img_erodida)

# função principal que gere fluxo da aplicação
def main() -> None:
    # pede para escolher entre carregar imagem de ficheiro ou usar webcam
    # .strip() remove espaços em branco à volta da entrada
    escolha_input = input("Enter image path or press Enter to use webcam: ").strip()

    # cria janela 'controlos' e trackbar para ajuste do limiar
    # feito antes de carregar imagem ou iniciar webcam,
    # para que janela de controlos esteja sempre disponível
    cv2.namedWindow('Controlos', cv2.WINDOW_AUTOSIZE) # autosize ajusta janela ao conteúdo
    # cria trackbar 'limiar' na janela 'controlos'
    # varia de 127 (valor inicial) a 255 (valor máximo)
    # 'ao_mudar_limiar': callback chamada quando valor muda
    cv2.createTrackbar('Limiar', 'Controlos', 127, 255, ao_mudar_limiar)

    if not escolha_input: # se entrada vazia (utilizador pressionou enter), usa webcam
        cap = cv2.VideoCapture(0) # inicializa captura da webcam (índice 0 é geralmente a padrão)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            cv2.destroyWindow('Controlos') # fecha janela de controlos se webcam falhar
            return
        
        print("Webcam started. Press 'q' to quit.")
        
        # nomes das janelas para exibir resultados do processamento da webcam
        window_names_webcam = ["Original", "Binary", "Contours", "Dilated", "Eroded"]
        for name in window_names_webcam:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL) # window_normal permite redimensionar janela
        
        try:
            while True:
                ret, frame = cap.read() # lê frame da webcam
                if not ret: # se 'ret' false, leitura do frame falhou
                    print("Error: Failed to capture frame from webcam.")
                    break
                
                frame = cv2.flip(frame, 1) # inverte frame horizontalmente (efeito espelho)
                cv2.imshow("Original", frame) # mostra frame original da webcam
                # processa e mostra diferentes vistas do frame
                # para webcam, frame original usado para processamento e desenhar contornos
                processar_e_mostrar_frame(frame, frame)
                                
                # verifica se tecla 'q' foi pressionada para sair do loop
                # cv2.waitkey(1) espera 1ms por tecla. & 0xff é máscara para garantir
                # compatibilidade entre sistemas (especialmente em 64 bits)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # bloco executado sempre ao sair do try (normalmente ou por exceção),
            # garantindo que recursos são libertados
            print("Releasing webcam and closing windows...")
            cap.release() # liberta objeto da webcam
            # fecha todas as janelas de visualização específicas da webcam
            for name in window_names_webcam:
                # verifica se janela ainda existe/visível antes de tentar destruí-la, para evitar erros
                if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow(name)
            # fecha janela de controlos
            if cv2.getWindowProperty('Controlos', cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow('Controlos')
            # cv2.destroyAllWindows() # pode ser usado como catch-all, mas fechar individualmente é mais controlado
    
    else: # se utilizador forneceu caminho, tenta carregar imagem
        img_original_carregada = cv2.imread(escolha_input)
        if img_original_carregada is None: # se imagem não pôde ser carregada (ex: caminho inválido)
            print(f"Error: Image not found or could not be loaded from path: {escolha_input}")
            cv2.destroyWindow('Controlos') # fecha janela de controlos
            return
        
        print(f"Image '{escolha_input}' loaded.")
        print("Adjust threshold in 'Controls' window. Press any key in an OpenCV window to process and exit.")
        
        # nomes das janelas para modo de imagem estática
        window_names_static_img = ["Original Static", "Binary", "Contours", "Dilated", "Eroded"]
        for name in window_names_static_img:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        
        # mostra imagem original carregada
        cv2.imshow("Original Static", img_original_carregada)

        # loop para permitir ajuste do trackbar e reprocessamento de imagem estática
        # sai quando qualquer tecla é pressionada
        while True:
            # importante fazer cópia da imagem original carregada para cada processamento,
            # pois funções de processamento (especialmente de desenho) podem modificar imagem
            img_para_processar_copia = img_original_carregada.copy()
            # processa e mostra diferentes vistas da imagem
            # imagem original carregada usada como base para desenhar contornos
            processar_e_mostrar_frame(img_para_processar_copia, img_original_carregada)

            # espera por tecla por 30ms
            # se tecla pressionada, key não será 255 (comum para "nenhuma tecla")
            key_press = cv2.waitKey(30) & 0xFF 
            if key_press != 255:
                print(f"Key pressed ({key_press}), exiting static image mode.")
                break
        
        # fecha todas as janelas ao sair do modo de imagem estática
        print("Closing windows...")
        for name in window_names_static_img:
            if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow(name)
        if cv2.getWindowProperty('Controlos', cv2.WND_PROP_VISIBLE) >= 1:
             cv2.destroyWindow('Controlos')
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
