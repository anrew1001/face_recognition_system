"""
Базовый класс для всех экранов GUI.
"""

from abc import ABC, abstractmethod
import customtkinter as ctk
from config.colors import DARK_BG


class BaseScreen(ctk.CTkFrame, ABC):
    """
    Абстрактный базовый класс для экранов приложения.
    Все экраны должны наследоваться от этого класса.
    """

    def __init__(self, parent, main_window):
        """
        Args:
            parent: Родительский виджет
            main_window: Ссылка на главное окно приложения (MainWindow)
        """
        super().__init__(parent, fg_color=DARK_BG)
        self.main_window = main_window

    def show(self):
        """
        Показывает экран. Вызывается при переключении на этот экран.
        Переопределите этот метод, если нужна дополнительная логика при показе.
        """
        self.pack(fill="both", expand=True)
        self.lift()

    def hide(self):
        """
        Скрывает экран. Вызывается при переключении на другой экран.
        Переопределите этот метод, если нужна дополнительная логика при скрытии.
        """
        self.pack_forget()

    def cleanup(self):
        """
        Очистка ресурсов экрана.
        Переопределите этот метод, если нужно освободить ресурсы
        (например, остановить видеопоток, закрыть соединения и т.д.).
        """
        pass
