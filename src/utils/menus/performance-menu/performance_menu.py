import os
import psutil
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
import numpy as np
import threading
from typing import List, Dict, Tuple, Optional
import random  # para gerar dados simulados de ia

# tema visual (similar ao strg-launcher para consistência)
THEME = {
    'bg': "#0a0a16",         # fundo escuro azulado
    'fg': "#e6e6ff",         # texto claro
    'accent': "#00dcff",     # destaque azul neon
    'accent_dark': "#00a0c8",# destaque escurecido
    'btn': "#1e1e32",        # botão escuro
    'hover': "#323250"       # botão hover
}

# configurações de fontes e tamanhos
FONT_CONFIG = {
    'title_size': 20,
    'label_size': 14,
    'value_size': 12,
    'axis_label_size': 12,
    'title_plot_size': 16,
    'tick_size': 10,
    'button_size': 12,
    'message_size': 12
}

class PerformanceGUI:
    def __init__(self, root=None):
        # inicialização do menu de performance
        self.metrics_history: Dict[str, List[float]] = {
            'cpu_percent': [],
            'memory_percent': [],
            'disk_usage': [],
            'response_times': [],
            'swap_usage_percent': [],
            'disk_read_bps': [],      # bytes lidos do disco por segundo
            'disk_write_bps': [],     # bytes escritos no disco por segundo
            'net_sent_bps': [],       # bytes enviados pela rede por segundo
            'net_recv_bps': []       # bytes recebidos pela rede por segundo
        }
        self.timestamps: List[datetime] = []
        self.start_time = datetime.now()
        self.running = False
        self.update_thread: Optional[threading.Thread] = None
        self.update_interval = 1.0  # segundos entre cada atualização
        self.max_history = 100  # número máximo de pontos de dados para guardar
        self.current_tab = 0  # aba atual selecionada
        
        # inicializa contadores para cálculo de taxas
        self.last_disk_io_counters = psutil.disk_io_counters() # para i/o de disco
        self.last_net_io_counters = psutil.net_io_counters() # para i/o de rede
        self.printed_warnings = set() # para evitar spam de warnings na consola
        
        # criação da interface gráfica
        self.own_window = root is None
        if self.own_window:
            self.root = tk.Tk()
            self.root.title("Performance Monitor")
            self.root.geometry("1280x800")  # aumenta tamanho da janela
            self.root.configure(bg=THEME['bg'])
        else:
            self.root = root

        # configuração global matplotlib para aumentar tamanho das fontes
        plt.rc('font', size=FONT_CONFIG['label_size'])
        plt.rc('axes', titlesize=FONT_CONFIG['title_plot_size'])
        plt.rc('axes', labelsize=FONT_CONFIG['axis_label_size'])
        plt.rc('xtick', labelsize=FONT_CONFIG['tick_size'])
        plt.rc('ytick', labelsize=FONT_CONFIG['tick_size'])
        plt.rc('legend', fontsize=FONT_CONFIG['label_size'])

        self.create_widgets()
        self.start_real_time_updates()
        
        if self.own_window:
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
            self.root.mainloop()

    def on_close(self):
        # para updates antes de fechar janela
        self.stop_real_time_updates()
        if self.own_window:
            self.root.destroy()

    def create_widgets(self):
        # criação do layout principal
        self.main_frame = tk.Frame(self.root, bg=THEME['bg'], padx=20, pady=20)  # aumenta padding
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # cabeçalho
        header_frame = tk.Frame(self.main_frame, bg=THEME['bg'])
        header_frame.pack(fill=tk.X, pady=(0, 20))  # aumenta espaçamento
        
        title_lbl = tk.Label(
            header_frame, 
            text="Performance Monitor", 
            font=("Arial", FONT_CONFIG['title_size'], "bold"),  # fonte maior
            fg=THEME['accent'],
            bg=THEME['bg']
        )
        title_lbl.pack(side=tk.LEFT)
        
        # métricas atuais
        self.metrics_frame = tk.Frame(self.main_frame, bg=THEME['bg'])
        self.metrics_frame.pack(fill=tk.X, pady=15)  # aumenta espaçamento
        
        # criação dos indicadores de métricas
        self.create_metric_indicator("CPU Usage", "cpu_percent")
        self.create_metric_indicator("Memory Usage", "memory_percent")
        self.create_metric_indicator("Disk Usage", "disk_usage")
        
        # criação do notebook (sistema de abas)
        style = ttk.Style()
        style.configure("TNotebook", background=THEME['bg'])
        style.configure("TNotebook.Tab", background=THEME['btn'], foreground=THEME['fg'], 
                        padding=[15, 5], font=('Arial', FONT_CONFIG['label_size']))  # abas maiores
        style.map("TNotebook.Tab", 
                  background=[("selected", THEME['accent_dark'])],
                  foreground=[("selected", THEME['fg'])])
        
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=15)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
        # aba de recursos do sistema
        self.system_frame = tk.Frame(self.notebook, bg=THEME['bg'])
        self.notebook.add(self.system_frame, text="System Resources")
        
        # configuração dos gráficos em cada aba
        self.setup_system_plots()
        
        # barra inferior com botões
        bottom_frame = tk.Frame(self.main_frame, bg=THEME['bg'])
        bottom_frame.pack(fill=tk.X, pady=(20, 0))  # aumenta espaçamento
        
        # botões de ações
        btn_export = self.create_button(bottom_frame, "Export Chart", self.save_graph)
        btn_export.pack(side=tk.RIGHT, padx=10)  # aumenta espaçamento
        
        btn_reset = self.create_button(bottom_frame, "Reset Data", self.reset_data)
        btn_reset.pack(side=tk.RIGHT, padx=10)  # aumenta espaçamento

        # informações de tempo de execução
        self.runtime_label = tk.Label(
            bottom_frame, 
            text="Runtime: 00:00:00",
            font=("Arial", FONT_CONFIG['label_size']),  # fonte maior
            fg=THEME['fg'],
            bg=THEME['bg']
        )
        self.runtime_label.pack(side=tk.LEFT, padx=10)  # aumenta espaçamento

    def on_tab_changed(self, event):
        # controla mudança de abas e atualiza interface
        self.current_tab = self.notebook.index(self.notebook.select())
        self.update_gui()

    def create_button(self, parent, text, command):
        # cria botões estilizados
        btn = tk.Button(
            parent,
            text=text,
            font=("Arial", FONT_CONFIG['button_size']),  # fonte maior
            fg=THEME['fg'],
            bg=THEME['btn'],
            activebackground=THEME['hover'],
            activeforeground=THEME['fg'],
            relief=tk.RAISED,
            padx=15,  # aumenta padding
            pady=8,   # aumenta padding
            command=command
        )
        return btn
        
    def create_metric_indicator(self, name, metric_key):
        # cria indicador de métrica com barra de progresso
        frame = tk.Frame(self.metrics_frame, bg=THEME['bg'], padx=10, pady=10)  # aumenta padding
        frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        label = tk.Label(
            frame, 
            text=name,
            font=("Arial", FONT_CONFIG['label_size']),  # fonte maior
            fg=THEME['fg'],
            bg=THEME['bg']
        )
        label.pack(anchor="w")
        
        # estilo para barra de progresso
        style = ttk.Style()
        style.theme_use('default')
        style.configure(
            f"{metric_key}.Horizontal.TProgressbar",
            troughcolor=THEME['bg'],
            bordercolor=THEME['bg'],
            background=THEME['accent'],
            lightcolor=THEME['accent'],
            darkcolor=THEME['accent_dark'],
            thickness=20  # barra mais grossa
        )
        
        progressbar = ttk.Progressbar(
            frame, 
            style=f"{metric_key}.Horizontal.TProgressbar",
            orient="horizontal", 
            length=300,  # barra mais longa
            mode="determinate"
        )
        progressbar.pack(fill=tk.X, pady=4)
        
        value_label = tk.Label(
            frame, 
            text="0.0%",
            font=("Arial", FONT_CONFIG['value_size']),  # fonte maior
            fg=THEME['accent'],
            bg=THEME['bg']
        )
        value_label.pack(anchor="e")
        
        # armazenar referências para atualização
        setattr(self, f"{metric_key}_bar", progressbar)
        setattr(self, f"{metric_key}_label", value_label)

    def setup_system_plots(self):
        # configura gráficos na aba de recursos do sistema
        system_fig = Figure(figsize=(14, 12), dpi=100, facecolor=THEME['bg']) # figsize ajustado para 3x3
        self.system_canvas = FigureCanvasTkAgg(system_fig, master=self.system_frame)
        self.system_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # subplots para recursos do sistema (grid 3x3)
        self.sys_ax1 = system_fig.add_subplot(331)  # cpu
        self.sys_ax2 = system_fig.add_subplot(332)  # memória
        self.sys_ax3 = system_fig.add_subplot(333)  # uso de disco %
        self.sys_ax4 = system_fig.add_subplot(334)  # uso de swap %
        self.sys_ax5 = system_fig.add_subplot(335)  # leitura disco bps
        self.sys_ax6 = system_fig.add_subplot(336)  # escrita disco bps
        self.sys_ax7 = system_fig.add_subplot(337)  # net sent bps
        self.sys_ax8 = system_fig.add_subplot(338)  # net recv bps
        self.sys_ax9 = system_fig.add_subplot(339)  # tempo de resposta
        
        # estilização dos plots
        axes_list = [self.sys_ax1, self.sys_ax2, self.sys_ax3, self.sys_ax4, 
                     self.sys_ax5, self.sys_ax6, self.sys_ax7, self.sys_ax8, self.sys_ax9]

        for ax in axes_list: 
            ax.set_facecolor(THEME['bg'])
            ax.tick_params(colors=THEME['fg'], which='both', labelsize=FONT_CONFIG['tick_size'])
            ax.xaxis.label.set_color(THEME['fg'])
            ax.xaxis.label.set_fontsize(FONT_CONFIG['axis_label_size'])
            ax.yaxis.label.set_color(THEME['fg']) 
            ax.yaxis.label.set_fontsize(FONT_CONFIG['axis_label_size'])
            for spine in ax.spines.values():
                spine.set_color(THEME['accent_dark'])
            ax.grid(True, alpha=0.3, color=THEME['accent_dark'])
        
        # títulos e rótulos
        self.sys_ax1.set_title('CPU Usage', color=THEME['accent'], fontsize=FONT_CONFIG['title_plot_size'])
        self.sys_ax1.set_ylabel('(%)', color=THEME['fg'])
        
        self.sys_ax2.set_title('Memory Usage', color=THEME['accent'], fontsize=FONT_CONFIG['title_plot_size'])
        self.sys_ax2.set_ylabel('(%)', color=THEME['fg'])
        
        self.sys_ax3.set_title('Disk Usage %', color=THEME['accent'], fontsize=FONT_CONFIG['title_plot_size'])
        self.sys_ax3.set_ylabel('(%)', color=THEME['fg'])
        
        self.sys_ax4.set_title('Swap Usage %', color=THEME['accent'], fontsize=FONT_CONFIG['title_plot_size'])
        self.sys_ax4.set_ylabel('(%)', color=THEME['fg'])

        self.sys_ax5.set_title('Disk Read', color=THEME['accent'], fontsize=FONT_CONFIG['title_plot_size'])
        self.sys_ax5.set_ylabel('(B/s)', color=THEME['fg'])

        self.sys_ax6.set_title('Disk Write', color=THEME['accent'], fontsize=FONT_CONFIG['title_plot_size'])
        self.sys_ax6.set_ylabel('(B/s)', color=THEME['fg'])
        
        self.sys_ax7.set_title('Network Sent', color=THEME['accent'], fontsize=FONT_CONFIG['title_plot_size'])
        self.sys_ax7.set_xlabel('Time (minutes)', color=THEME['fg'])
        self.sys_ax7.set_ylabel('(B/s)', color=THEME['fg'])

        self.sys_ax8.set_title('Network Received', color=THEME['accent'], fontsize=FONT_CONFIG['title_plot_size'])
        self.sys_ax8.set_xlabel('Time (minutes)', color=THEME['fg'])
        self.sys_ax8.set_ylabel('(B/s)', color=THEME['fg'])

        self.sys_ax9.set_title('Response Times', color=THEME['accent'], fontsize=FONT_CONFIG['title_plot_size'])
        self.sys_ax9.set_xlabel('Time (minutes)', color=THEME['fg'])
        self.sys_ax9.set_ylabel('(seconds)', color=THEME['fg'])

        # remove x-tick labels para plots não inferiores
        for ax in [self.sys_ax1, self.sys_ax2, self.sys_ax3, self.sys_ax4, self.sys_ax5, self.sys_ax6]:
            ax.set_xticklabels([]) 

        system_fig.tight_layout(pad=2.0) # ajustar padding
        
        # linhas iniciais vazias
        self.cpu_line, = self.sys_ax1.plot([], [], color=THEME['accent'], linewidth=2.0)
        self.mem_line, = self.sys_ax2.plot([], [], color='magenta', linewidth=2.0)
        self.disk_line, = self.sys_ax3.plot([], [], color='yellow', linewidth=2.0)
        self.swap_line, = self.sys_ax4.plot([], [], color='cyan', linewidth=2.0)
        self.disk_read_line, = self.sys_ax5.plot([], [], color='#ff6347', linewidth=2.0)
        self.disk_write_line, = self.sys_ax6.plot([], [], color='#32cd32', linewidth=2.0)
        self.net_sent_line, = self.sys_ax7.plot([], [], color='#ff8c00', linewidth=2.0) # darkorange
        self.net_recv_line, = self.sys_ax8.plot([], [], color='#1e90ff', linewidth=2.0) # dodgerblue
        self.resp_line, = self.sys_ax9.plot([], [], color='#00fa9a', linewidth=2.0) # mediumspringgreen

    def update_metrics(self, response_time: float = None) -> None:
        # atualiza métricas do sistema
        self.metrics_history['cpu_percent'].append(psutil.cpu_percent())
        self.metrics_history['memory_percent'].append(psutil.virtual_memory().percent)
        self.metrics_history['disk_usage'].append(psutil.disk_usage('/').percent)
        try:
            self.metrics_history['swap_usage_percent'].append(psutil.swap_memory().percent)
        except Exception as e: 
            self.metrics_history['swap_usage_percent'].append(np.nan) 
            if 'swap_usage_percent' not in self.printed_warnings:
                print(f"Aviso: Não foi possível obter métricas de swap: {e}")
                self.printed_warnings.add('swap_usage_percent')

        # cálculo de i/o de disco por segundo
        current_disk_io = psutil.disk_io_counters()
        if self.last_disk_io_counters:
            read_bps = (current_disk_io.read_bytes - self.last_disk_io_counters.read_bytes) / self.update_interval
            write_bps = (current_disk_io.write_bytes - self.last_disk_io_counters.write_bytes) / self.update_interval
            self.metrics_history['disk_read_bps'].append(max(0, read_bps))
            self.metrics_history['disk_write_bps'].append(max(0, write_bps))
        else:
            self.metrics_history['disk_read_bps'].append(0)
            self.metrics_history['disk_write_bps'].append(0)
        self.last_disk_io_counters = current_disk_io

        # cálculo de i/o de rede por segundo
        current_net_io = psutil.net_io_counters()
        if self.last_net_io_counters:
            sent_bps = (current_net_io.bytes_sent - self.last_net_io_counters.bytes_sent) / self.update_interval
            recv_bps = (current_net_io.bytes_recv - self.last_net_io_counters.bytes_recv) / self.update_interval
            self.metrics_history['net_sent_bps'].append(max(0, sent_bps))
            self.metrics_history['net_recv_bps'].append(max(0, recv_bps))
        else:
            self.metrics_history['net_sent_bps'].append(0)
            self.metrics_history['net_recv_bps'].append(0)
        self.last_net_io_counters = current_net_io

        if response_time is not None:
            self.metrics_history['response_times'].append(response_time)
        else:
            pass 
        
        self.timestamps.append(datetime.now())
        
        # limita tamanho do histórico para evitar excesso de memória
        if len(self.timestamps) > self.max_history:
            for key in self.metrics_history: # iterar sobre chaves que existem
                if self.metrics_history[key]: # garante que lista não está vazia antes de fatiar
                    self.metrics_history[key] = self.metrics_history[key][-self.max_history:]
            self.timestamps = self.timestamps[-self.max_history:]

    def start_real_time_updates(self) -> None:
        # inicia atualizações em tempo real
        self.running = True
        
        def update_loop():
            while self.running:
                self.update_metrics()
                # agenda atualização da interface na thread principal
                self.root.after(0, self.update_gui)
                time.sleep(self.update_interval)
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()

    def stop_real_time_updates(self) -> None:
        # para atualizações em tempo real
        self.running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)

    def update_gui(self) -> None:
        # atualiza elementos da interface gráfica com métricas atuais
        if not self.metrics_history['cpu_percent']:
            return  # sem dados para atualizar
            
        # atualiza barras de progresso e labels
        for metric_key in ['cpu_percent', 'memory_percent', 'disk_usage']:
            value = self.metrics_history[metric_key][-1]
            progressbar = getattr(self, f"{metric_key}_bar")
            label = getattr(self, f"{metric_key}_label")
            
            progressbar['value'] = value
            label.config(text=f"{value:.1f}%")
        
        # atualiza gráficos da aba atual
        self.update_plots_for_current_tab()
        
        # atualiza tempo de execução
        runtime = datetime.now() - self.start_time
        hours = int(runtime.total_seconds() // 3600)
        minutes = int((runtime.total_seconds() % 3600) // 60)
        seconds = int(runtime.total_seconds() % 60)
        self.runtime_label.config(text=f"Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def update_plots_for_current_tab(self) -> None:
        # atualiza apenas gráficos da aba atual para economizar recursos
        if not self.metrics_history['cpu_percent'] or not self.timestamps:
            return

        time_points = [(t - self.start_time).total_seconds() / 60 for t in self.timestamps]

        current_tab_text = self.notebook.tab(self.notebook.select(), "text")

        if current_tab_text == "System Resources":
            self.update_system_plots(time_points)
        # condições para "gpu & ai", "model performance", "advanced ai metrics" removidas

    def update_system_plots(self, time_points):
        # atualiza gráficos da aba de recursos do sistema
        self.cpu_line.set_data(time_points, self.metrics_history['cpu_percent'])
        self.mem_line.set_data(time_points, self.metrics_history['memory_percent'])
        self.disk_line.set_data(time_points, self.metrics_history['disk_usage'])
        self.swap_line.set_data(time_points, self.metrics_history['swap_usage_percent'])
        self.disk_read_line.set_data(time_points, self.metrics_history['disk_read_bps'])
        self.disk_write_line.set_data(time_points, self.metrics_history['disk_write_bps'])
        self.net_sent_line.set_data(time_points, self.metrics_history['net_sent_bps'])
        self.net_recv_line.set_data(time_points, self.metrics_history['net_recv_bps'])
        
        current_response_times = self.metrics_history.get('response_times', [])
        if current_response_times:
            resp_points_len = len(current_response_times)
            resp_points = time_points[-resp_points_len:] if len(time_points) >= resp_points_len else time_points[:resp_points_len]
            self.resp_line.set_data(resp_points, current_response_times)
        else:
            self.resp_line.set_data([], []) 
        
        plot_configs = [
            (self.sys_ax1, self.cpu_line, False), (self.sys_ax2, self.mem_line, False),
            (self.sys_ax3, self.disk_line, False), (self.sys_ax4, self.swap_line, False),
            (self.sys_ax5, self.disk_read_line, True), (self.sys_ax6, self.disk_write_line, True),
            (self.sys_ax7, self.net_sent_line, True), (self.sys_ax8, self.net_recv_line, True),
            (self.sys_ax9, self.resp_line, True) 
        ]

        for ax, line, is_rate_or_time in plot_configs:
            if len(line.get_xdata()) > 0:
                ax.set_xlim(0, max(1, max(line.get_xdata())))
                
                ydata = line.get_ydata()
                valid_ydata = [val for val in ydata if not np.isnan(val)]
                
                if not valid_ydata:
                    ymin, ymax = (0, 1) if is_rate_or_time else (0, 100)
                else:
                    if is_rate_or_time:
                        ymin = min(valid_ydata) if min(valid_ydata) < 0 else 0 # permite ymin negativo se dados o tiverem, senão 0
                        ymax = max(1, max(valid_ydata) * 1.1) 
                    else: # percentagens
                        ymin = 0
                        ymax = max(100, max(valid_ydata)) 
                ax.set_ylim(ymin, ymax)
        
        self.system_canvas.draw_idle()

    def update_plots(self) -> None:
        # mantém este método para compatibilidade com código anterior
        # ou para atualizar todos os plots se necessário.
        # no entanto, com lógica de atualização por aba, esta pode não ser mais necessária.
        # self.update_plots_for_current_tab() # chama atualização focada na aba
        pass

    def save_graph(self) -> None:
        # salva gráfico atual como imagem com data e hora no nome
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # apenas aba "system" permanece
        current_tab_name = "system" 
        save_path = os.path.join(os.getcwd(), f"{current_tab_name}_performance_chart_{timestamp}.png")
        
        try:
            # apenas lógica para aba "system" (current_tab == 0) é necessária
            if self.notebook.index(self.notebook.select()) == 0: # verifica se aba "system resources" está selecionada
                self.system_canvas.figure.savefig(save_path, facecolor=self.system_canvas.figure.get_facecolor(), dpi=300)
                # mostra mensagem temporária
                msg_label = tk.Label(self.root, text=f"Chart saved to {save_path}", fg="white", bg="green", font=("Arial", FONT_CONFIG['message_size']))
                msg_label.place(relx=0.5, rely=0.05, anchor="center")
                self.root.after(3000, msg_label.destroy)
            else:
                # caso improvável de outra aba existir e ser selecionada, ou erro
                print("Selected tab cannot be exported or does not exist.")

        except Exception as e:
            # mostra mensagem de erro
            msg_label = tk.Label(self.root, text=f"Error saving chart: {e}", fg="white", bg="red", font=("Arial", FONT_CONFIG['message_size']))
            msg_label.place(relx=0.5, rely=0.05, anchor="center")
            self.root.after(3000, msg_label.destroy)

    def reset_data(self) -> None:
        # reinicia todas as métricas coletadas
        for key in self.metrics_history:
            self.metrics_history[key] = []
        self.timestamps = []
        self.start_time = datetime.now()

# função para compatibilidade com código anterior
class PerformanceMenu:
    def __init__(self):
        self.gui = None
        
    def display_menu(self):
        # cria gui quando menu for exibido
        if self.gui is None:
            self.gui = PerformanceGUI()
        else:
            # se já existe, apenas mostra janela
            self.gui.root.deiconify()
            
    def update_metrics(self, response_time=None):
        # passa métricas para gui se existir
        if self.gui:
            self.gui.update_metrics(response_time)

if __name__ == '__main__':
    # executa versão gui diretamente
    app = PerformanceGUI() 