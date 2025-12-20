"""
Главное окно приложения для системы распознавания лиц.
"""

import customtkinter as ctk
from typing import Dict, Optional
from config.colors import DARK_BG


class MainWindow(ctk.CTk):
    """
    Главное окно приложения с навигацией между экранами.
    """

    def __init__(self):
        super().__init__()

        # Настройка окна
        self.title("Face Recognition System")
        self.geometry("800x600")

        # Устанавливаем темную тему
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Устанавливаем цвет фона
        self.configure(fg_color=DARK_BG)

        # Словарь для хранения экранов
        self.screens: Dict[str, ctk.CTkFrame] = {}

        # Текущий активный экран
        self.current_screen: Optional[str] = None

        # Контейнер для экранов
        self.screen_container = ctk.CTkFrame(self, fg_color=DARK_BG)
        self.screen_container.pack(fill="both", expand=True)

    def add_screen(self, name: str, screen: ctk.CTkFrame):
        """
        Добавляет экран в окно.

        Args:
            name: Имя экрана для навигации
            screen: Объект экрана (наследник BaseScreen)
        """
        self.screens[name] = screen

    def show_screen(self, screen_name: str):
        """
        Показывает указанный экран и скрывает текущий.

        Args:
            screen_name: Имя экрана для отображения
        """
        if screen_name not in self.screens:
            raise ValueError(f"Экран '{screen_name}' не найден")

        # Скрываем текущий экран
        if self.current_screen and self.current_screen in self.screens:
            current = self.screens[self.current_screen]
            if hasattr(current, 'hide'):
                current.hide()
            else:
                current.pack_forget()

        # Показываем новый экран
        new_screen = self.screens[screen_name]
        if hasattr(new_screen, 'show'):
            new_screen.show()
        else:
            new_screen.pack(fill="both", expand=True)

        self.current_screen = screen_name

        # Обновляем геометрию для корректного отображения
        self.update_idletasks()

    def run(self):
        """
        Запускает главный цикл приложения.
        """
        self.mainloop()