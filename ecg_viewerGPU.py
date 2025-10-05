import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy import signal
import struct
import time

# GPU Support - глобална променлива
GPU_AVAILABLE = False
cp = None

try:
    import cupy as cp
    from cupyx.scipy import signal as cusignal

    GPU_AVAILABLE = True
    print("✓ GPU (CUDA) Support Enabled")
except ImportError:
    print("✗ GPU Support Not Available - Install cupy for CUDA acceleration")


class ECGViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Universal ECG Viewer with GPU Acceleration")
        self.root.geometry("1400x900")

        # Параметри (автоматично детектирани или по подразбиране)
        self.sampling_rate = 1000  # Hz (може да се променя)
        self.num_leads = 12
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                           'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        # Данни
        self.ecg_data = None
        self.ecg_data_raw = None
        self.current_position = 0
        self.window_duration = 10
        self.current_gain = 1.0
        self.current_file = None
        self.file_info = {}

        # GPU Settings
        global GPU_AVAILABLE
        self.use_gpu = tk.BooleanVar(value=GPU_AVAILABLE)
        self.gpu_device = None
        if GPU_AVAILABLE and cp is not None:
            try:
                self.gpu_device = cp.cuda.Device(0)
                gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
                print(f"GPU Device: {gpu_name}")
            except Exception as e:
                print(f"GPU initialization failed: {e}")
                GPU_AVAILABLE = False
                self.use_gpu.set(False)

        self.create_widgets()

    def create_widgets(self):
        # Меню бар
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Отвори ECG файл", command=self.show_load_dialog)
        file_menu.add_command(label="Презареди сегмент...", command=self.reload_segment)
        file_menu.add_separator()
        file_menu.add_command(label="Експорт в CSV", command=self.export_csv)
        file_menu.add_command(label="Запази графика", command=self.save_plot)
        file_menu.add_separator()
        file_menu.add_command(label="Изход", command=self.root.quit)

        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Настройки", menu=settings_menu)
        settings_menu.add_command(label="Конфигурация на отвеждания", command=self.configure_leads)
        settings_menu.add_command(label="Честота на семплиране", command=self.configure_sampling_rate)

        # Контролен панел
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control_frame, text="Позиция (сек):").pack(side=tk.LEFT, padx=5)
        self.position_var = tk.StringVar(value="0")
        self.position_entry = ttk.Entry(control_frame, textvariable=self.position_var, width=10)
        self.position_entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Скок", command=self.jump_to_position).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="◄◄ Назад", command=self.prev_window).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="►► Напред", command=self.next_window).pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Прозорец (сек):").pack(side=tk.LEFT, padx=20)
        self.window_var = tk.StringVar(value="10")
        window_combo = ttk.Combobox(control_frame, textvariable=self.window_var,
                                    values=['5', '10', '30', '60'], width=5)
        window_combo.pack(side=tk.LEFT, padx=5)
        window_combo.bind('<<ComboboxSelected>>', lambda e: self.update_plot())

        # Gain контрол
        ttk.Label(control_frame, text="Мащаб:").pack(side=tk.LEFT, padx=20)
        self.gain_var = tk.StringVar(value="1.0")
        gain_combo = ttk.Combobox(control_frame, textvariable=self.gain_var,
                                  values=['0.1', '0.5', '1.0', '2.0', '5.0', '10.0'], width=5)
        gain_combo.pack(side=tk.LEFT, padx=5)
        gain_combo.bind('<<ComboboxSelected>>', lambda e: self.apply_gain())

        ttk.Button(control_frame, text="🔍 Auto Scale", command=self.auto_scale).pack(side=tk.LEFT, padx=5)

        # Филтър контроли
        self.filter_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Филтър", variable=self.filter_var,
                        command=self.update_plot).pack(side=tk.LEFT, padx=20)

        # GPU контрол
        if GPU_AVAILABLE and cp is not None:
            ttk.Checkbutton(control_frame, text="🚀 GPU", variable=self.use_gpu,
                            command=self.toggle_gpu).pack(side=tk.LEFT, padx=5)

        # Информационен панел
        info_frame = ttk.Frame(self.root, padding="10")
        info_frame.pack(side=tk.TOP, fill=tk.X)

        self.info_label = ttk.Label(info_frame, text="Няма зареден файл",
                                    font=('Arial', 10))
        self.info_label.pack(side=tk.LEFT)

        # Heart Rate индикатор
        self.hr_label = ttk.Label(info_frame, text="HR: -- bpm",
                                  font=('Arial', 11, 'bold'), foreground='red')
        self.hr_label.pack(side=tk.RIGHT, padx=20)

        # GPU Status
        if GPU_AVAILABLE and cp is not None:
            self.gpu_label = ttk.Label(info_frame, text="🚀 GPU: Active",
                                       font=('Arial', 10, 'bold'), foreground='green')
            self.gpu_label.pack(side=tk.RIGHT, padx=10)
        else:
            self.gpu_label = None

        # График
        self.figure = Figure(figsize=(14, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        toolbar.update()

        # Статус бар
        self.status_var = tk.StringVar(value="Готов")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def toggle_gpu(self):
        """Превключва GPU режим"""
        if not GPU_AVAILABLE or cp is None or self.gpu_label is None:
            return

        if self.use_gpu.get():
            self.gpu_label.config(text="🚀 GPU: Active", foreground='green')
            self.status_var.set("GPU ускорение активирано")
        else:
            self.gpu_label.config(text="GPU: Inactive", foreground='gray')
            self.status_var.set("CPU режим")

    def configure_leads(self):
        """Диалог за конфигуриране на отвеждания"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Конфигурация на отвеждания")
        dialog.geometry("400x500")
        dialog.transient(self.root)

        ttk.Label(dialog, text="Брой отвеждания:", font=('Arial', 10, 'bold')).pack(pady=10)

        num_leads_var = tk.StringVar(value=str(self.num_leads))
        ttk.Radiobutton(dialog, text="3 отвеждания", variable=num_leads_var, value="3").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(dialog, text="5 отвеждания", variable=num_leads_var, value="5").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(dialog, text="12 отвеждания (стандартно)", variable=num_leads_var, value="12").pack(anchor=tk.W,
                                                                                                            padx=20)
        ttk.Radiobutton(dialog, text="Друго:", variable=num_leads_var, value="custom").pack(anchor=tk.W, padx=20)

        custom_frame = ttk.Frame(dialog)
        custom_frame.pack(fill=tk.X, padx=40)
        custom_var = tk.StringVar(value="12")
        ttk.Entry(custom_frame, textvariable=custom_var, width=10).pack(side=tk.LEFT)

        def apply_config():
            try:
                if num_leads_var.get() == "custom":
                    num = int(custom_var.get())
                else:
                    num = int(num_leads_var.get())

                if num < 1 or num > 32:
                    messagebox.showerror("Грешка", "Броят отвеждания трябва да е между 1 и 32")
                    return

                self.num_leads = num
                # Генерираме имена на отвеждания
                if num == 12:
                    self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                                       'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
                elif num == 3:
                    self.lead_names = ['I', 'II', 'III']
                elif num == 5:
                    self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL']
                else:
                    self.lead_names = [f'Ch{i + 1}' for i in range(num)]

                messagebox.showinfo("Успех", f"Конфигурирани {num} отвеждания")
                dialog.destroy()

            except ValueError:
                messagebox.showerror("Грешка", "Невалидно число")

        ttk.Button(dialog, text="Приложи", command=apply_config).pack(pady=20)

    def configure_sampling_rate(self):
        """Диалог за промяна на честотата на семплиране"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Честота на семплиране")
        dialog.geometry("300x200")
        dialog.transient(self.root)

        ttk.Label(dialog, text="Честота (Hz):", font=('Arial', 10, 'bold')).pack(pady=10)

        rate_var = tk.StringVar(value=str(self.sampling_rate))

        for rate in [125, 250, 500, 1000, 2000, 4000]:
            ttk.Radiobutton(dialog, text=f"{rate} Hz", variable=rate_var, value=str(rate)).pack(anchor=tk.W, padx=20)

        def apply_rate():
            try:
                new_rate = int(rate_var.get())
                self.sampling_rate = new_rate
                messagebox.showinfo("Успех", f"Честота зададена на {new_rate} Hz")
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Грешка", "Невалидна честота")

        ttk.Button(dialog, text="Приложи", command=apply_rate).pack(pady=20)

    def show_load_dialog(self):
        """Показва диалог за избор на файл и опции за зареждане"""
        filename = filedialog.askopenfilename(
            title="Изберете ECG файл",
            filetypes=[("Binary files", "*.BIN *.bin *.dat *.DAT"),
                       ("All files", "*.*")]
        )

        if not filename:
            return

        try:
            import os
            file_size = os.path.getsize(filename)
            file_size_mb = file_size / (1024 * 1024)

            # Опит за определяне на общата продължителност
            estimated_samples = (file_size - 512) // 2 // self.num_leads
            estimated_duration_hours = estimated_samples / self.sampling_rate / 3600

            # Показваме диалог с опции
            dialog = tk.Toplevel(self.root)
            dialog.title("Опции за зареждане")
            dialog.geometry("500x450")
            dialog.transient(self.root)
            dialog.grab_set()

            # Информация
            info_frame = ttk.LabelFrame(dialog, text="Информация за файла", padding=10)
            info_frame.pack(fill=tk.X, padx=10, pady=10)

            ttk.Label(info_frame, text=f"Файл: {os.path.basename(filename)}").pack(anchor=tk.W)
            ttk.Label(info_frame, text=f"Размер: {file_size_mb:.1f} MB").pack(anchor=tk.W)
            ttk.Label(info_frame, text=f"Приблизителна продължителност: {estimated_duration_hours:.1f} часа").pack(
                anchor=tk.W)
            ttk.Label(info_frame, text=f"Конфигурация: {self.num_leads} отвеждания @ {self.sampling_rate}Hz").pack(
                anchor=tk.W)

            # GPU Info
            if GPU_AVAILABLE and cp is not None and self.use_gpu.get():
                ttk.Label(info_frame, text="🚀 GPU ускорение: Активно", foreground='green').pack(anchor=tk.W)

            # Опции за зареждане
            options_frame = ttk.LabelFrame(dialog, text="Изберете какво да заредите", padding=10)
            options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            load_option = tk.StringVar(value="full")

            ttk.Radiobutton(options_frame, text="📄 Целият файл",
                            variable=load_option, value="full").pack(anchor=tk.W, pady=5)

            ttk.Radiobutton(options_frame, text="⏱️ Първите 15 минути",
                            variable=load_option, value="15min").pack(anchor=tk.W, pady=5)

            ttk.Radiobutton(options_frame, text="⏱️ Първите 30 минути",
                            variable=load_option, value="30min").pack(anchor=tk.W, pady=5)

            ttk.Radiobutton(options_frame, text="⏱️ Първият 1 час",
                            variable=load_option, value="1hour").pack(anchor=tk.W, pady=5)

            ttk.Radiobutton(options_frame, text="⏱️ Първите 2 часа",
                            variable=load_option, value="2hours").pack(anchor=tk.W, pady=5)

            ttk.Radiobutton(options_frame, text="⏱️ Първите 6 часа",
                            variable=load_option, value="6hours").pack(anchor=tk.W, pady=5)

            # Custom опция
            custom_frame = ttk.Frame(options_frame)
            custom_frame.pack(fill=tk.X, pady=10)

            ttk.Radiobutton(custom_frame, text="✏️ Персонализиран диапазон:",
                            variable=load_option, value="custom").pack(side=tk.LEFT)

            custom_frame2 = ttk.Frame(options_frame)
            custom_frame2.pack(fill=tk.X, padx=20)

            ttk.Label(custom_frame2, text="От (мин):").pack(side=tk.LEFT, padx=5)
            start_var = tk.StringVar(value="0")
            ttk.Entry(custom_frame2, textvariable=start_var, width=10).pack(side=tk.LEFT, padx=5)

            ttk.Label(custom_frame2, text="До (мин):").pack(side=tk.LEFT, padx=5)
            end_var = tk.StringVar(value="60")
            ttk.Entry(custom_frame2, textvariable=end_var, width=10).pack(side=tk.LEFT, padx=5)

            # Бутони
            button_frame = ttk.Frame(dialog)
            button_frame.pack(fill=tk.X, padx=10, pady=10)

            def on_load():
                option = load_option.get()
                start_min = 0
                end_min = None

                if option == "full":
                    start_min, end_min = 0, None
                elif option == "15min":
                    start_min, end_min = 0, 15
                elif option == "30min":
                    start_min, end_min = 0, 30
                elif option == "1hour":
                    start_min, end_min = 0, 60
                elif option == "2hours":
                    start_min, end_min = 0, 120
                elif option == "6hours":
                    start_min, end_min = 0, 360
                elif option == "custom":
                    try:
                        start_min = float(start_var.get())
                        end_min = float(end_var.get())
                    except:
                        messagebox.showerror("Грешка", "Невалидни стойности за диапазон")
                        return

                dialog.destroy()
                self.load_file(filename, start_min, end_min)

            ttk.Button(button_frame, text="✓ Зареди", command=on_load).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="✗ Отказ", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

            dialog.wait_window()

        except Exception as e:
            messagebox.showerror("Грешка", f"Грешка при анализ на файла:\n{str(e)}")

    def load_file(self, filename, start_min=0, end_min=None):
        """Зарежда файл с опционален диапазон и GPU ускорение"""
        try:
            start_time = time.time()
            self.status_var.set("Зареждане на файл...")
            self.root.update()

            self.current_file = filename

            # Четене на файла
            with open(filename, 'rb') as f:
                data = f.read()

            file_size = len(data)

            # Опит за разпознаване на header
            header_sizes = [0, 64, 128, 256, 512, 1024]
            best_header = 0

            for header_size in header_sizes:
                try:
                    payload_size = file_size - header_size
                    bytes_per_sample = 2
                    total_values = payload_size // bytes_per_sample

                    if total_values % self.num_leads == 0:
                        best_header = header_size
                        break
                except:
                    continue

            offset = best_header
            payload = data[offset:]

            # Преобразуване в 16-bit integers
            num_samples = len(payload) // 2
            raw_data = np.frombuffer(payload[:num_samples * 2], dtype=np.int16)

            # Пресъстояване в канали
            samples_per_lead = len(raw_data) // self.num_leads
            full_data = raw_data[:samples_per_lead * self.num_leads].reshape(
                samples_per_lead, self.num_leads
            )

            # Изчисляваме диапазона за зареждане
            start_sample = int(start_min * 60 * self.sampling_rate)
            if end_min is not None:
                end_sample = int(end_min * 60 * self.sampling_rate)
                end_sample = min(end_sample, samples_per_lead)
            else:
                end_sample = samples_per_lead

            # Извличаме желания диапазон
            self.ecg_data = full_data[start_sample:end_sample].copy()

            # Запазваме информация
            self.file_info = {
                'total_duration': samples_per_lead / self.sampling_rate,
                'loaded_start': start_sample / self.sampling_rate,
                'loaded_end': end_sample / self.sampling_rate,
                'loaded_duration': (end_sample - start_sample) / self.sampling_rate
            }

            # Преобразуване с GPU ако е активирано
            if GPU_AVAILABLE and cp is not None and self.use_gpu.get():
                self.status_var.set("GPU обработка...")
                self.root.update()
                self.ecg_data = self._process_data_gpu(self.ecg_data)
            else:
                self.ecg_data = self._process_data_cpu(self.ecg_data)

            # Запазваме raw версия
            self.ecg_data_raw = self.ecg_data.copy()

            load_time = time.time() - start_time
            duration_sec = len(self.ecg_data) / self.sampling_rate
            duration_hours = duration_sec / 3600
            total_hours = self.file_info['total_duration'] / 3600

            loaded_range = f"{start_min:.0f}-{end_min if end_min else 'край'}мин" if end_min else "пълен"

            gpu_info = "🚀 GPU" if (GPU_AVAILABLE and cp is not None and self.use_gpu.get()) else "CPU"

            self.info_label.config(
                text=f"Файл: {filename.split('/')[-1]} | "
                     f"Зареден: {duration_hours:.2f}ч ({loaded_range}) от {total_hours:.1f}ч | "
                     f"Samples: {len(self.ecg_data):,} | {self.sampling_rate}Hz | "
                     f"{gpu_info} | Време: {load_time:.2f}s"
            )

            self.current_position = 0
            self.update_plot()

            self.root.after(500, self.auto_scale)

            self.status_var.set(f"Файлът е зареден успешно за {load_time:.2f}s")

        except Exception as e:
            messagebox.showerror("Грешка", f"Грешка при зареждане:\n{str(e)}")
            self.status_var.set("Грешка при зареждане")

    def _process_data_gpu(self, data):
        """Обработва данните с GPU"""
        if cp is None:
            return self._process_data_cpu(data)

        try:
            # Прехвърляме към GPU
            gpu_data = cp.array(data, dtype=cp.float32)

            # Премахваме baseline offset
            for i in range(self.num_leads):
                baseline = cp.median(gpu_data[:, i])
                gpu_data[:, i] -= baseline

            # Нормализираме amplitude
            gpu_data = gpu_data / 200.0

            # Връщаме на CPU
            return cp.asnumpy(gpu_data)
        except Exception as e:
            print(f"GPU processing failed: {e}, falling back to CPU")
            return self._process_data_cpu(data)

    def _process_data_cpu(self, data):
        """Обработва данните с CPU"""
        data = data.astype(np.float32)

        for i in range(self.num_leads):
            baseline = np.median(data[:, i])
            data[:, i] -= baseline

        data = data / 200.0
        return data

    def reload_segment(self):
        """Презарежда различен сегмент от текущия файл"""
        if self.current_file is None:
            messagebox.showwarning("Внимание", "Първо заредете файл")
            return

        self.show_load_dialog_for_current_file()

    def show_load_dialog_for_current_file(self):
        """Показва диалог за презареждане на сегмент"""
        if self.current_file is None:
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Презареди сегмент")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()

        total_hours = self.file_info.get('total_duration', 0) / 3600
        total_minutes = self.file_info.get('total_duration', 0) / 60

        info_frame = ttk.LabelFrame(dialog, text="Информация", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(info_frame, text=f"Обща продължителност: {total_hours:.1f}ч ({total_minutes:.0f}мин)").pack(
            anchor=tk.W)
        ttk.Label(info_frame,
                  text=f"Текущо зареден: {self.file_info.get('loaded_start', 0) / 60:.0f}-{self.file_info.get('loaded_end', 0) / 60:.0f}мин").pack(
            anchor=tk.W)

        options_frame = ttk.LabelFrame(dialog, text="Избери нов диапазон", padding=10)
        options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(options_frame, text="От (минути):").pack(anchor=tk.W, pady=5)
        start_var = tk.StringVar(value="0")
        ttk.Entry(options_frame, textvariable=start_var, width=15).pack(anchor=tk.W, padx=20)

        ttk.Label(options_frame, text="До (минути) - остави празно за край:").pack(anchor=tk.W, pady=5)
        end_var = tk.StringVar(value="")
        ttk.Entry(options_frame, textvariable=end_var, width=15).pack(anchor=tk.W, padx=20)

        quick_frame = ttk.Frame(options_frame)
        quick_frame.pack(fill=tk.X, pady=10)
        ttk.Label(quick_frame, text="Бързи опции:").pack(side=tk.LEFT)

        def quick_load(start, end):
            start_var.set(str(start))
            end_var.set(str(end) if end else "")

        ttk.Button(quick_frame, text="15мин", command=lambda: quick_load(0, 15), width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_frame, text="1час", command=lambda: quick_load(0, 60), width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_frame, text="Всичко", command=lambda: quick_load(0, None), width=8).pack(side=tk.LEFT, padx=2)

        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        def on_load():
            try:
                start = float(start_var.get())
                end_str = end_var.get().strip()
                end = float(end_str) if end_str else None

                if end is not None and end <= start:
                    messagebox.showerror("Грешка", "Крайната позиция трябва да е по-голяма от началната")
                    return

                dialog.destroy()
                self.load_file(self.current_file, start, end)

            except ValueError:
                messagebox.showerror("Грешка", "Невалидни стойности")

        ttk.Button(button_frame, text="Зареди", command=on_load).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Отказ", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def update_plot(self):
        if self.ecg_data is None:
            return

        try:
            self.window_duration = int(self.window_var.get())
        except:
            self.window_duration = 10

        window_samples = self.window_duration * self.sampling_rate
        start_sample = self.current_position
        end_sample = min(start_sample + window_samples, len(self.ecg_data))

        data_segment = self.ecg_data[start_sample:end_sample]

        # Прилагаме филтър
        if self.filter_var.get() and len(data_segment) > 100:
            try:
                data_segment = self.filter_ecg_signal(data_segment)
            except:
                pass

        time = np.arange(len(data_segment)) / self.sampling_rate

        self.figure.clear()

        bg_color = 'white'
        grid_major_color = '#FF9999'
        grid_minor_color = '#FFE5E5'
        signal_color = 'black'

        # Динамичен layout базиран на броя отвеждания
        if self.num_leads <= 6:
            rows, cols = self.num_leads, 1
        elif self.num_leads <= 12:
            rows, cols = 6, 2
        else:
            rows, cols = int(np.ceil(self.num_leads / 4)), 4

        for i in range(self.num_leads):
            ax = self.figure.add_subplot(rows, cols, i + 1, facecolor=bg_color)

            ax.set_xlim(time[0], time[-1])
            ax.grid(True, which='major', linestyle='-', linewidth=1.0,
                    color=grid_major_color, alpha=0.8)
            ax.minorticks_on()
            ax.grid(True, which='minor', linestyle='-', linewidth=0.5,
                    color=grid_minor_color, alpha=0.6)

            from matplotlib.ticker import MultipleLocator
            ax.xaxis.set_major_locator(MultipleLocator(0.2))
            ax.xaxis.set_minor_locator(MultipleLocator(0.04))
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))

            ax.plot(time, data_segment[:, i], signal_color, linewidth=1.2,
                    antialiased=True, solid_capstyle='round')

            ax.set_ylabel(f'{self.lead_names[i]}', fontsize=10,
                          fontweight='bold', rotation=0, ha='right', va='center')

            lead_data = data_segment[:, i]
            if len(lead_data) > 0:
                data_std = np.std(lead_data)
                data_mean = np.mean(lead_data)

                if data_std > 0.01:
                    y_range = max(data_std * 4, 1.0)
                    ax.set_ylim(data_mean - y_range, data_mean + y_range)
                else:
                    ax.set_ylim(-1.5, 1.5)

            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.tick_params(labelsize=8)

            if i >= self.num_leads - cols:
                ax.set_xlabel('Време (s)', fontsize=9)
            else:
                ax.set_xticklabels([])

            for spine in ax.spines.values():
                spine.set_edgecolor('#CCCCCC')
                spine.set_linewidth(1)

        loaded_time_info = ""
        if self.file_info:
            abs_time_sec = self.file_info['loaded_start'] + (self.current_position / self.sampling_rate)
            loaded_time_info = f" | Абс. време: {abs_time_sec / 60:.1f}мин"

        self.figure.suptitle(
            f'{self.num_leads}-Lead ECG - Position: {self.current_position / self.sampling_rate:.1f}s '
            f'({(self.current_position / self.sampling_rate) / 60:.1f}min){loaded_time_info}',
            fontsize=11, fontweight='bold'
        )

        self.figure.tight_layout()
        self.canvas.draw()

        hr = self.calculate_heart_rate(data_segment)
        if hr:
            self.hr_label.config(text=f"HR: {hr} bpm")
        else:
            self.hr_label.config(text="HR: -- bpm")

        self.position_var.set(f"{self.current_position / self.sampling_rate:.1f}")

    def next_window(self):
        if self.ecg_data is None:
            return

        window_samples = self.window_duration * self.sampling_rate
        self.current_position = min(
            self.current_position + window_samples,
            len(self.ecg_data) - window_samples
        )
        self.update_plot()

    def prev_window(self):
        if self.ecg_data is None:
            return

        window_samples = self.window_duration * self.sampling_rate
        self.current_position = max(0, self.current_position - window_samples)
        self.update_plot()

    def jump_to_position(self):
        if self.ecg_data is None:
            return

        try:
            pos_sec = float(self.position_var.get())
            pos_sample = int(pos_sec * self.sampling_rate)
            self.current_position = max(0, min(pos_sample, len(self.ecg_data) - 1))
            self.update_plot()
        except ValueError:
            messagebox.showerror("Грешка", "Невалидна позиция")

    def export_csv(self):
        if self.ecg_data is None:
            messagebox.showwarning("Внимание", "Няма заредени данни")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not filename:
            return

        try:
            self.status_var.set("Експортиране в CSV...")
            self.root.update()

            max_rows = 100000
            export_data = self.ecg_data[:max_rows]

            header = ','.join(['Time(s)'] + self.lead_names)
            time_col = np.arange(len(export_data)) / self.sampling_rate

            with open(filename, 'w') as f:
                f.write(header + '\n')
                for i, t in enumerate(time_col):
                    row = [f"{t:.3f}"] + [f"{export_data[i, j]:.6f}"
                                          for j in range(self.num_leads)]
                    f.write(','.join(row) + '\n')

            self.status_var.set("CSV експортиран успешно")
            messagebox.showinfo("Успех", f"Данните са експортирани в:\n{filename}")

        except Exception as e:
            messagebox.showerror("Грешка", f"Грешка при експорт:\n{str(e)}")
            self.status_var.set("Грешка при експорт")

    def save_plot(self):
        if self.ecg_data is None:
            messagebox.showwarning("Внимание", "Няма заредени данни")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"),
                       ("All files", "*.*")]
        )

        if filename:
            self.figure.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Успех", f"Графиката е запазена в:\n{filename}")

    def apply_gain(self):
        if self.ecg_data_raw is None:
            return

        try:
            gain = float(self.gain_var.get())
            self.current_gain = gain
            self.ecg_data = self.ecg_data_raw * gain
            self.update_plot()
            self.status_var.set(f"Мащаб приложен: {gain}x")
        except ValueError:
            pass

    def auto_scale(self):
        if self.ecg_data is None:
            return

        sample_size = min(10000, len(self.ecg_data))
        sample_data = self.ecg_data[:sample_size]

        amplitudes = []
        for i in range(self.num_leads):
            lead_data = sample_data[:, i]
            amp = np.percentile(lead_data, 95) - np.percentile(lead_data, 5)
            if amp > 0:
                amplitudes.append(amp)

        if amplitudes:
            median_amp = np.median(amplitudes)
            target_amp = 1.5
            suggested_gain = target_amp / median_amp if median_amp > 0 else 1.0

            standard_gains = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            suggested_gain = min(standard_gains, key=lambda x: abs(x - suggested_gain))

            self.gain_var.set(str(suggested_gain))
            self.apply_gain()

    def filter_ecg_signal(self, data):
        """Филтрира ECG сигнал с GPU или CPU"""
        if GPU_AVAILABLE and cp is not None and self.use_gpu.get() and len(data) > 10000:
            return self._filter_gpu(data)
        else:
            return self._filter_cpu(data)

    def _filter_gpu(self, data):
        """GPU-ускорено филтриране"""
        if cp is None:
            return self._filter_cpu(data)

        try:
            # Прехвърляме към GPU
            gpu_data = cp.array(data)
            filtered_data = cp.zeros_like(gpu_data)

            nyquist = self.sampling_rate / 2
            low_freq = 0.5 / nyquist
            high_freq = 40.0 / nyquist

            # Bandpass filter
            b, a = signal.butter(4, [low_freq, high_freq], btype='band')
            b_gpu = cp.array(b)
            a_gpu = cp.array(a)

            # Notch filter
            notch_freq = 50.0 / nyquist
            b_notch, a_notch = signal.iirnotch(notch_freq, 30)
            b_notch_gpu = cp.array(b_notch)
            a_notch_gpu = cp.array(a_notch)

            for i in range(self.num_leads):
                # За GPU filtering използваме scipy на CPU, но паралелно
                lead_cpu = cp.asnumpy(gpu_data[:, i])
                filtered_lead = signal.filtfilt(b, a, lead_cpu)
                filtered_lead = signal.filtfilt(b_notch, a_notch, filtered_lead)
                filtered_data[:, i] = cp.array(filtered_lead)

            return cp.asnumpy(filtered_data)
        except Exception as e:
            print(f"GPU filtering failed: {e}, falling back to CPU")
            return self._filter_cpu(data)

    def _filter_cpu(self, data):
        """CPU филтриране"""
        filtered_data = np.zeros_like(data)

        nyquist = self.sampling_rate / 2
        low_freq = 0.5 / nyquist
        high_freq = 40.0 / nyquist

        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        notch_freq = 50.0 / nyquist
        b_notch, a_notch = signal.iirnotch(notch_freq, 30)

        for i in range(self.num_leads):
            filtered_lead = signal.filtfilt(b, a, data[:, i])
            filtered_lead = signal.filtfilt(b_notch, a_notch, filtered_lead)
            filtered_data[:, i] = filtered_lead

        return filtered_data

    def calculate_heart_rate(self, data_segment):
        """Изчислява heart rate"""
        try:
            # Използваме Lead II ако съществува, иначе първото отвеждане
            lead_idx = min(1, self.num_leads - 1)
            lead_data = data_segment[:, lead_idx]

            if not self.filter_var.get():
                lead_filtered = self.filter_ecg_signal(data_segment)[:, lead_idx]
            else:
                lead_filtered = lead_data

            threshold = np.std(lead_filtered) * 0.6
            min_distance = int(0.4 * self.sampling_rate)

            peaks, _ = signal.find_peaks(lead_filtered,
                                         height=threshold,
                                         distance=min_distance)

            if len(peaks) > 1:
                rr_intervals = np.diff(peaks) / self.sampling_rate
                mean_rr = np.mean(rr_intervals)
                heart_rate = 60.0 / mean_rr

                return int(heart_rate)
            else:
                return None
        except:
            return None


if __name__ == "__main__":
    root = tk.Tk()
    app = ECGViewer(root)
    root.mainloop()