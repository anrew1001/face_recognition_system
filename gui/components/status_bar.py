"""
Компонент статус-бара для отображения информации о системе.
"""

import customtkinter as ctk
from config.colors import STATUS_BG, LIGHT_TEXT, GREEN_SUCCESS, CYAN_ACCENT


class StatusBar(ctk.CTkFrame):
    """
    Статус-бар для отображения информации о состоянии системы.
    Показывает: FPS, количество лиц, количество идентификаций в БД,
    статус шифрования и название модели.
    """

    def __init__(self, parent):
        """
        Args:
            parent: Родительский виджет
        """
        super().__init__(parent, fg_color=STATUS_BG, height=40)

        # Создаем метки для различных параметров
        self._create_status_labels()

        # Устанавливаем начальные значения
        self.update_fps(0)
        self.update_faces_count(0)
        self.update_db_count(0)
        self.update_encryption_status(True)
        self.update_model_name("buffalo_l")

    def _create_status_labels(self):
        """Создает метки статус-бара."""
        # Контейнер для центрирования элементов
        self.inner_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.inner_frame.pack(pady=5, padx=10, fill="x")

        # FPS
        self.fps_label = ctk.CTkLabel(
            self.inner_frame,
            text="FPS: 0",
            font=("Roboto", 12),
            text_color=LIGHT_TEXT
        )
        self.fps_label.pack(side="left", padx=10)

        # Количество лиц на экране
        self.faces_label = ctk.CTkLabel(
            self.inner_frame,
            text="👤 Лиц: 0",
            font=("Roboto", 12),
            text_color=LIGHT_TEXT
        )
        self.faces_label.pack(side="left", padx=10)

        # Количество идентификаций в БД
        self.db_count_label = ctk.CTkLabel(
            self.inner_frame,
            text="📊 База: 0",
            font=("Roboto", 12),
            text_color=CYAN_ACCENT
        )
        self.db_count_label.pack(side="left", padx=10)

        # Статус шифрования
        self.encryption_label = ctk.CTkLabel(
            self.inner_frame,
            text="🔒 Шифрование: Вкл",
            font=("Roboto", 12),
            text_color=GREEN_SUCCESS
        )
        self.encryption_label.pack(side="left", padx=10)

        # Название модели
        self.model_label = ctk.CTkLabel(
            self.inner_frame,
            text="Модель: buffalo_l",
            font=("Roboto", 12),
            text_color=LIGHT_TEXT
        )
        self.model_label.pack(side="right", padx=10)

    def update_fps(self, fps: float):
        """
        Обновляет отображение FPS.

        Args:
            fps: Значение FPS
        """
        self.fps_label.configure(text=f"FPS: {fps:.1f}")

    def update_faces_count(self, count: int):
        """
        Обновляет количество обнаруженных лиц.

        Args:
            count: Количество лиц
        """
        self.faces_label.configure(text=f"👤 Лиц: {count}")

    def update_db_count(self, count: int):
        """
        Обновляет количество идентификаций в базе данных.

        Args:
            count: Количество записей в БД
        """
        self.db_count_label.configure(text=f"📊 База: {count}")

    def update_encryption_status(self, enabled: bool):
        """
        Обновляет статус шифрования.

        Args:
            enabled: True если шифрование включено, False иначе
        """
        status_text = "Вкл" if enabled else "Выкл"
        color = GREEN_SUCCESS if enabled else LIGHT_TEXT
        self.encryption_label.configure(
            text=f"🔒 Шифрование: {status_text}",
            text_color=color
        )

    def update_model_name(self, model_name: str):
        """
        Обновляет название используемой модели.

        Args:
            model_name: Название модели
        """
        self.model_label.configure(text=f"Модель: {model_name}")
