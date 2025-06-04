import os
import subprocess
from pynput import keyboard
import tkinter as tk
from tkinter import font as tkfont
import shutil
import random # para o campo de estrelas
import math # para animações

# Configuração do tema (cores em formato RGB para melhor manipulação)
THEME = {
    'bg': (10, 10, 22),         # Fundo escuro 
    'fg': (230, 230, 255),      # Texto claro
    'accent': (0, 220, 255),    # Destaque azul neon 
    'accent_dark': (0, 160, 200),# Destaque escurecido para cliques
    'btn': (30, 30, 50),        # Botão escuro
    'hover': (50, 50, 80),      # Botão hover
    'glow': (0, 220, 255, 150)  # Brilho com alfa 
}

# Conversão para hexadecimal (Tkinter não suporta alfa em cores hexadecimais diretamente)
def rgb_to_hex(rgb):
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"

# Mapeamento dos programas (refatorado para clareza)
# Ordem reorganizada para melhor agrupamento lógico
PROGRAMS_CONFIG = [
    {
        'key': keyboard.Key.f1,
        'path': 'src/main/main.py',
        'name': 'Reconhecimento de Gestos',
        'icon': '👋'
    },
    {
        'key': keyboard.Key.f2, 
        'path': 'src/main/word_recognition/word_recognition_app.py',
        'name': 'Reconhecimento de Palavras',
        'icon': '🗣️'
    },
    {
        'key': keyboard.Key.f3,
        'path': 'src/utils/mouse-control-hand/mouse_control.py',
        'name': 'Controlo do Cursor',
        'icon': '🖱️'
    },
    {
        'key': keyboard.Key.f4,
        'path': 'src/utils/volume-control-hand/main.py',
        'name': 'Controlo do Volume',
        'icon': '🔊'
    },
    {
        'key': keyboard.Key.f5,
        'path': 'src/utils/face-recon/main.py',
        'name': 'Reconhecimento Facial',
        'icon': '👤'
    },
    {
        'key': keyboard.Key.f6,
        'path': 'src/utils/binary-vision/binary_vision.py',
        'name': 'Visão Binária',
        'icon': '👁️'
    },
    {
        'key': keyboard.Key.f7,
        'path': 'src/utils/3d-hand-viewer/python/hand_detection.py',
        'name': 'Visualizador 3D',
        'icon': '🖐️'
    },
    {
        'key': keyboard.Key.f8,
        'path': 'src/utils/menus/performance-menu/performance_menu.py',
        'name': 'Menu de Performance',
        'icon': '⚡'
    }
]

class AnimatedButton(tk.Canvas):
    def __init__(self, master, text, command, icon=None, width=360, height=200):
        super().__init__(master, width=width, height=height,
                         bg=rgb_to_hex(THEME['bg']), highlightthickness=0)
        # comando a ser executado quando o botão é pressionado
        self.command = command
        # largura do canvas do botão
        self.width = width
        # altura do canvas do botão
        self.height = height
        # texto exibido no botão
        self.text = text
        # ícone opcional exibido no botão
        self.icon = icon
        # flag para indicar se o cursor do rato está sobre o botão
        self.is_hovered = False
        # estado da animação, usado para efeitos como pulsação (atualmente não usado ativamente para brilho complexo)
        self.animation_state = 0

        self._draw_button_base()
        self._draw_content()
        self._setup_bindings()

    def _draw_button_base(self, color_key='btn'):
        # desenha a forma base do botão com cantos arredondados
        # utiliza uma chave de cor para determinar a cor de preenchimento e da borda
        self.delete("base")
        radius = 25 # raio dos cantos arredondados
        # pontos para criar um polígono com cantos arredondados.
        # a técnica é definir os pontos de início e fim dos arcos e as linhas retas
        points = [
            radius, 0, self.width - radius, 0,  # linha superior
            self.width, radius, self.width, self.height - radius,  # canto superior direito e linha direita
            self.width - radius, self.height, radius, self.height,  # canto inferior direito e linha inferior
            0, self.height - radius, 0, radius  # canto inferior esquerdo e linha esquerda
        ]
        self.create_polygon(points, fill=rgb_to_hex(THEME[color_key]),
                            # a cor da borda muda se o rato estiver sobre o botão
                            outline=rgb_to_hex(THEME['accent' if self.is_hovered else 'btn']),
                            # a largura da borda também muda com o hover
                            width=3 if self.is_hovered else 2, smooth=True, tags="base")

    def _draw_content(self):
        # desenha o ícone e o texto dentro do botão
        self.delete("content")
        # ícone (se existir) é centralizado na parte superior do botão
        if self.icon:
            self.create_text(
                self.width / 2, self.height * 0.38, # posicionamento do ícone
                text=self.icon, font=('Arial', 42, 'bold'),
                fill=rgb_to_hex(THEME['accent']), tags="content"
            )
        # texto é centralizado na parte inferior do botão
        self.create_text(
            self.width / 2, self.height * 0.80, # posicionamento do texto
            text=self.text, font=('Arial', 12, 'bold'),
            fill=rgb_to_hex(THEME['fg']),
            width=self.width - 50, # largura máxima do texto para permitir quebra de linha
            justify='center', tags="content"
        )

    def _setup_bindings(self):
        # configura os bindings de eventos do rato para interatividade
        # <Enter>: cursor entra na área do botão
        self.bind("<Enter>", self._on_enter)
        # <Leave>: cursor sai da área do botão
        self.bind("<Leave>", self._on_leave)
        # <Button-1>: botão esquerdo do rato é pressionado
        self.bind("<Button-1>", self._on_click_press)
        # <ButtonRelease-1>: botão esquerdo do rato é solto
        self.bind("<ButtonRelease-1>", self._on_click_release)

    def _on_enter(self, event=None):
        # chamado quando o cursor entra na área do botão
        self.is_hovered = True
        self._draw_button_base(color_key='hover') # redesenha a base com cor de hover
        self._draw_content() # redesenha o conteúdo (pode ser necessário se o estado afetar o texto/ícone)
        self._animate_glow() #inicia a animação de brilho (atualmente sutil).

    def _on_leave(self, event=None):
        # chamado quando o cursor sai da área do botão
        self.is_hovered = False
        self.delete("glow_effect") # remove explicitamente qualquer efeito de brilho
        self._draw_button_base(color_key='btn') # restaura a cor base do botão
        self._draw_content()

    def _animate_glow(self):
        #  responsável por animar um efeito de brilho no botão quando o cursor está sobre ele
        # a implementação atual foca-se no brilho da borda, controlado em _draw_button_base
        if not self.is_hovered:
            self.delete("glow_effect") # garante que não há brilho se não estiver em hover
            return

        self.delete("glow_effect") # limpa efeitos de brilho anteriores para redesenhar 
    
        # a animação de brilho principal é controlada pela mudança da cor e largura da borda em _draw_button_base
        # o código abaixo para um brilho pulsante mais complexo foi comentado
        # para simplificar ou porque não era o efeito desejado
        # Exemplo de lógica de brilho pulsante (atualmente não ativa):
        # radius = 25 + 5 * math.sin(self.animation_state) # Pequena pulsação no raio.
        # self.create_oval(
        ##     self.width/2 - radius, self.height/2 - radius,
        #     self.width/2 + radius, self.height/2 + radius,
        #     outline=rgb_to_hex(THEME['accent']), width=2, tags="glow_effect"
        # )
        # self.animation_state += 0.3 # Incrementa o estado da animação
        # self.after(50, self._animate_glow) # Chama recursivamente para animação contínua
        pass # o brilho é atualmente gerido pela borda dinâmica no _on_enter/_on_leave

    def _on_click_press(self, event=None):
        # chamado quando o botão é pressionado
        self._draw_button_base(color_key='accent_dark') # muda a cor da base para feedback visual de clique
        self._draw_content()

    def _on_click_release(self, event=None):
        # chamado quando o botão do rato é solto
        if self.is_hovered: # se o rato ainda estiver sobre o botão, volta ao estado de hover
            self._draw_button_base(color_key='hover')
        else: # caso contrário, volta ao estado normal
            self._draw_button_base(color_key='btn')
        self._draw_content()
        
        # verifica se o clique ocorreu dentro dos limites do botão antes de executar o comando
        # isto é uma boa prática, especialmente para botões com formas não retangulares
        if 0 <= event.x <= self.width and 0 <= event.y <= self.height:
            self.command()


class StarryBackground(tk.Canvas):
    def __init__(self, master, num_stars=150, star_speed=0.5):
        super().__init__(master, bg=rgb_to_hex(THEME['bg']), highlightthickness=0)
        # número de estrelas a serem exibidas no fundo
        self.num_stars = num_stars
        # velocidade base do movimento das estrelas
        self.star_speed = star_speed
        # lista para armazenar os dados de cada estrela (ID, posição, tamanho, velocidade)
        self.stars = []
        # recria as estrelas quando o widget é redimensionado
        self.bind("<Configure>", self._resize_stars)
        self._create_stars()
        self._animate_stars()

    def _create_stars(self):
        # cria as estrelas iniciais no canvas
        # cada estrela tem uma posição, tamanho e cor aleatórios
        self.delete("star") # remove estrelas existentes antes de criar novas
        width = self.winfo_width() # obtém a largura atual do canvas
        height = self.winfo_height() # obtém a altura atual do canvas
        if width == 1 and height == 1:
             self.after(50, self._create_stars)
             return

        for _ in range(self.num_stars):
            x = random.uniform(0, width)
            y = random.uniform(0, height)
            size = random.uniform(1, 3) # tamanho aleatório para as estrelas
            
            # determina a cor da estrela com base num roll de brilho
            # estrelas mais brilhantes (cor de destaque) são mais raras
            brightness_roll = random.random()
            if brightness_roll < 0.1: # 10% de chance de ser muito brilhante
                color_rgb = THEME['accent']
            elif brightness_roll < 0.5: # 40% de chance de brilho médio
                color_rgb = THEME['fg']
            else: # 50% de chance de pouco brilho (cor mais escura)
                color_rgb = (
                    THEME['fg'][0] // 2,
                    THEME['fg'][1] // 2,
                    THEME['fg'][2] // 2,
                )
            # cria a oval que representa a estrela
            star = self.create_oval(x, y, x + size, y + size,
                                    fill=rgb_to_hex(color_rgb), outline="", tags="star")
            # armazena os dados da estrela para animação
            # as estrelas têm um movimento predominantemente descendente.
            self.stars.append({
                'id': star, 'x': x, 'y': y, 'size': size, 
                'vx': random.uniform(-self.star_speed, self.star_speed), # velocidade horizontal aleatória
                'vy': random.uniform(0.1, self.star_speed * 2) # velocidade vertical aleatória, maioritariamente para baixo
            })

    def _resize_stars(self, event=None):
        # chamado quando o canvas é redimensionado
        # limpa as estrelas existentes e cria novas para preencher o novo tamanho
        self.stars.clear()  
        self._create_stars()

    def _animate_stars(self):
        # anima o movimento das estrelas no canvas
        width = self.winfo_width()
        height = self.winfo_height()

        # se não houver estrelas e o canvas estiver dimensionado, cria-as
        if not self.stars and (width > 1 and height > 1) :
            self._create_stars()
            
        for star_data in self.stars:
            star_id = star_data['id']
            # atualiza a posição da estrela com base na sua velocidade
            star_data['x'] += star_data['vx']
            star_data['y'] += star_data['vy']

            # reposiciona as estrelas que saem dos limites da tela para criar um efeito de loop
            if star_data['y'] > height: # se sair por baixo, reaparece em cima
                star_data['y'] = 0
                star_data['x'] = random.uniform(0, width)
            elif star_data['y'] < 0: # se sair por cima (caso haja vx negativo), reaparece em baixo
                star_data['y'] = height
                star_data['x'] = random.uniform(0, width)
            
            if star_data['x'] > width: # se sair pela direita, reaparece à esquerda
                star_data['x'] = 0
            elif star_data['x'] < 0: # se sair pela esquerda, reaparece à direita
                star_data['x'] = width
            
            try:
                # move a estrela para a nova posição no canvas
                self.coords(star_id, star_data['x'], star_data['y'],
                            star_data['x'] + star_data['size'], star_data['y'] + star_data['size'])
            except tk.TclError: # trata o erro caso a estrela já não exista (pode acontecer durante redimensionamento rápido)
                pass # ignora o erro e continua.
        
        # agenda a próxima chamada a _animate_stars para criar um loop de animação
        # 50ms resulta em aproximadamente 20 FPS
        self.after(50, self._animate_stars)


class App:
    def __init__(self, root):
        self.root = root
        self._setup_window()
        self._create_ui()
        self._start_keyboard_listener()

    def _setup_window(self):
        # configura a janela principal da aplicação
        self.root.title("STRG") # define o título da janela
        self.root.geometry("1700x900") # define o tamanho inicial da janela
        # define a paleta de cores de fundo para a janela (pode não ser aplicado a todos os widgets)
        self.root.tk_setPalette(background=rgb_to_hex(THEME['bg']))

        # cria e exibe o fundo estrelado dinâmico
        self.star_bg = StarryBackground(self.root)
        self.star_bg.pack(fill=tk.BOTH, expand=True) # preenche todo o espaço disponível

        try: 
            # tenta centralizar a janela no ecrã
            # este comando é específico do Tk e pode não funcionar em todos os window managers (WMs
            self.root.eval('tk::PlaceWindow . center')
        except tk.TclError:
            pass # ignora silenciosamente se o comando falhar

    def _create_ui(self):
        # cria a interface gráfica do utilizador (GUI) sobre o fundo estrelado
        # o StarryBackground já foi configurado em _setup_window e empacotado

        # cabeçalho da aplicação
        header_font = tkfont.Font(family="Orbitron", size=36, weight="bold")
        header_label = tk.Label(self.star_bg, text="STRG", 
                                fg=rgb_to_hex(THEME['accent']), bg=rgb_to_hex(THEME['bg']), 
                                font=header_font)
        header_label.pack(pady=(50, 30)) # padding vertical (top: 50px, bottom: 30px)

        # frame para conter a grelha de botões, permitindo melhor controlo de layout e centralização
        grid_frame = tk.Frame(self.star_bg, bg=rgb_to_hex(THEME['bg']))
        grid_frame.pack(expand=True, fill=tk.BOTH, padx=50, pady=20) # expande e preenche com padding (left: 50px, right: 50px, top: 20px, bottom: 20px)

        num_programs = len(PROGRAMS_CONFIG)
        cols = 4 # define o número de colunas para a grelha de botões
        rows = math.ceil(num_programs / cols) # calcula o número de linhas necessárias

        for i, program_cfg in enumerate(PROGRAMS_CONFIG):
            row_idx = i // cols # calcula o índice da linha
            col_idx = i % cols  # calcula o índice da coluna
            
            # frame externo para cada botão, para controlar o preenchimento (sticky) na célula da grelha
            btn_outer_frame = tk.Frame(grid_frame, bg=rgb_to_hex(THEME['bg']))
            # usa padx e pady no grid para espaçamento entre as células dos botões
            btn_outer_frame.grid(row=row_idx, column=col_idx, padx=15, pady=15, sticky="nsew")
            
            button = AnimatedButton(
                btn_outer_frame, # o botão é filho do seu frame externo
                text=program_cfg['name'],
                command=lambda p=program_cfg: self._launch_program(p),
                icon=program_cfg['icon']
            )
            # faz o botão preencher o seu frame externo, expandindo-se
            button.pack(expand=True, fill=tk.BOTH)

        # configura as linhas e colunas do grid_frame para terem peso (weight)
        # isto permite que as células da grelha se expandam proporcionalmente quando a janela é redimensionada
        for r_idx in range(rows):
            grid_frame.grid_rowconfigure(r_idx, weight=1)
        for c_idx in range(cols):
            grid_frame.grid_columnconfigure(c_idx, weight=1)
        
        # rodapé com uma label de status ou instruções
        # self.status_label é usado em _launch_program, por isso precisa ser um atributo da classe
        self.status_label = tk.Label(
            self.star_bg, # adicionado ao star_bg para ficar sobre ele
            text="Sistema de Tradução e Reconhecimento de Gestos.", 
            font=('Consolas', 10),
            fg=rgb_to_hex(THEME['fg']), 
            bg=rgb_to_hex(THEME['bg'])
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(10,20), padx=10)

    def _launch_program(self, program_cfg):
        # inicia um programa especificado na configuração
        # abre o programa num novo terminal
        try:
            abs_path = os.path.abspath(program_cfg['path']) # obtém o caminho absoluto do script
            abs_dir = os.path.dirname(abs_path) # obtém o diretório do script
            terminal = self._detect_terminal() # detecta um terminal compatível
            # Adicionado para depuração
            print(f"DEBUG: Detected terminal: {terminal}")
            print(f"DEBUG: Script absolute path: {abs_path}")
            print(f"DEBUG: Script directory: {abs_dir}")

            if terminal and os.path.exists(abs_path):
                # atualiza a label de status para indicar que o programa está a iniciar
                self.status_label.config(text=f"A iniciar: {program_cfg['name']}...")
                # comando para abrir o terminal e executar o script python
                # aspas são usadas em abs_path para lidar com caminhos que possam ter espaços
                command_to_run = f"{terminal} -e 'python3 \"{abs_path}\"'"
                # Adicionado para depuração
                print(f"DEBUG: Executing command: {command_to_run}")
                subprocess.Popen(
                    command_to_run,
                    cwd=abs_dir, # define o diretório de trabalho para o script
                    shell=True   # necessário para interpretar o comando como uma string
                )
                # restaura a mensagem de status após 3 segundos
                self.root.after(3000, lambda: self.status_label.config(text="Sistema de Tradução e Reconhecimento de Gestos."))
            else:
                # atualiza a label de status se o terminal ou o script não forem encontrados
                error_msg = f"Erro: Terminal ou script não encontrado para {program_cfg['name']}."
                if not terminal:
                    error_msg += " (Terminal not detected)"
                if not os.path.exists(abs_path):
                    error_msg += f" (Script not found at {abs_path})"
                # Adicionado para depuração
                print(f"DEBUG: {error_msg}")
                self.status_label.config(text=error_msg)
        except Exception as e:
            # atualiza a label de status em caso de outros erros ao iniciar o programa
            error_msg = f"Erro ao iniciar: {str(e)}"
            # Adicionado para depuração
            print(f"DEBUG: Exception during launch: {error_msg}")
            self.status_label.config(text=error_msg)


    def _detect_terminal(self):
        # tenta detetar um emulador de terminal comum instalado no sistema
        for term_cmd in ['gnome-terminal', 'konsole', 'xfce4-terminal', 'terminator', 'xterm']:
            if shutil.which(term_cmd): # shutil.which verifica se o comando existe no PATH
                return term_cmd # retorna o primeiro terminal encontrado
        return None # retorna None se nenhum terminal for encontrado

    def _start_keyboard_listener(self):
        # inicia um listener de teclado para atalhos globais (F-keys).
        # mapeia as teclas configuradas para os programas correspondentes.
        self.keymap = {p['key']: p for p in PROGRAMS_CONFIG}
        
        # usa threading para o listener de teclado para não bloquear a thread principal da GUI (Tkinter)
        def listen():   
            # o listener é iniciado e junta-se à thread, mantendo-a viva enquanto o listener estiver ativo
            with keyboard.Listener(on_press=self._on_key_press) as listener:
                listener.join()

        import threading
        # cria e inicia a thread do listener como uma daemon thread
        # daemon threads são terminadas automaticamente quando o programa principal termina
        self.listener_thread = threading.Thread(target=listen, daemon=True)
        self.listener_thread.start()

    def _on_key_press(self, key):
        # callback chamado quando uma tecla é pressionada (pelo listener de teclado)
        # este método é executado na thread do listener, não na thread da GUI
        if key in self.keymap:
            # para interagir com a GUI (Tkinter) a partir de outra thread,
            # é necessário agendar a chamada do método _launch_program na thread principal da GUI
            # root.after(0, ...) agenda a chamada para ser executada o mais rápido possível
            self.root.after(0, self._launch_program, self.keymap[key])


if __name__ == '__main__':
    # ponto de entrada principal da aplicação
    root = tk.Tk() # cria a janela raiz do Tkinter
    
    # tenta verificar a existência de fontes específicas (Orbitron, Consolas)
    # se não encontradas, a aplicação continuará com fontes padrão
    try:
        tkfont.Font(family='Orbitron', size=1) # verifica se a fonte Orbitron está disponível
        tkfont.Font(family='Consolas', size=1) # verifica se a fonte Consolas está disponível
    except tk.TclError:
        pass # falha silenciosamente se as fontes não forem encontradas
    
    app = App(root) # cria a instância principal da aplicação
    root.mainloop() # inicia o loop de eventos do Tkinter, que mantém a janela aberta e responsiva