"""
Главное меню приложения.
"""

import customtkinter as ctk
from gui.screens.base_screen import BaseScreen
from gui.components.status_bar import StatusBar
from config.colors import (
    DARK_BG, CYAN_ACCENT, LIGHT_TEXT, TRANSPARENT
)


class MainMenuScreen(BaseScreen):
    """
    Экран главного меню с кнопками навигации.
    """

    def __init__(self, parent, main_window):
        super().__init__(parent, main_window)
        self._build_ui()

    def _build_ui(self):
        """Строит пользовательский интерфейс главного меню."""
        # Основной контейнер
        main_container = ctk.CTkFrame(self, fg_color=DARK_BG)
        main_container.pack(fill="both", expand=True)

        # Верхняя панель с кнопкой настроек
        self._create_top_panel(main_container)

        # Центральная область с кнопками
        self._create_center_panel(main_container)

        # Нижняя панель со статус-баром
        self._create_bottom_panel(main_container)

    def _create_top_panel(self, parent):
        """
        Создает верхнюю панель с кнопкой настроек.

        Args:
            parent: Родительский виджет
        """
        top_panel = ctk.CTkFrame(parent, fg_color="transparent", height=60)
        top_panel.pack(fill="x", padx=20, pady=(20, 0))
        top_panel.pack_propagate(False)

        # Кнопка настроек (справа)
        settings_button = ctk.CTkButton(
            top_panel,
            text="⚙️",
            font=("Roboto", 24),
            width=50,
            height=50,
            fg_color=TRANSPARENT,
            hover_color=CYAN_ACCENT,
            border_width=2,
            border_color=CYAN_ACCENT,
            command=self._on_settings_click
        )
        settings_button.pack(side="right")

    def _create_center_panel(self, parent):
        """
        Создает центральную панель с основными кнопками.

        Args:
            parent: Родительский виджет
        """
        center_panel = ctk.CTkFrame(parent, fg_color="transparent")
        center_panel.pack(fill="both", expand=True, padx=40, pady=20)

        # Кнопка распознавания (большая, акцентная)
        recognition_button = ctk.CTkButton(
            center_panel,
            text="👁️  РАСПОЗНАВАНИЕ",
            font=("Roboto", 28, "bold"),
            height=120,
            fg_color=CYAN_ACCENT,
            hover_color="#0096B8",
            text_color=LIGHT_TEXT,
            corner_radius=15,
            command=self._on_recognition_click
        )
        recognition_button.pack(fill="x", pady=(40, 20))

        # Кнопка регистрации (прозрачная с границей)
        registration_button = ctk.CTkButton(
            center_panel,
            text="➕  РЕГИСТРАЦИЯ",
            font=("Roboto", 20),
            height=80,
            fg_color=TRANSPARENT,
            hover_color=CYAN_ACCENT,
            text_color=LIGHT_TEXT,
            border_width=2,
            border_color=CYAN_ACCENT,
            corner_radius=15,
            command=self._on_registration_click
        )
        registration_button.pack(fill="x", pady=10)

        # Кнопка базы данных (прозрачная с границей)
        database_button = ctk.CTkButton(
            center_panel,
            text="📋  БАЗА ДАННЫХ",
            font=("Roboto", 20),
            height=80,
            fg_color=TRANSPARENT,
            hover_color=CYAN_ACCENT,
            text_color=LIGHT_TEXT,
            border_width=2,
            border_color=CYAN_ACCENT,
            corner_radius=15,
            command=self._on_database_click
        )
        database_button.pack(fill="x", pady=10)

    def _create_bottom_panel(self, parent):
        """
        Создает нижнюю панель со статус-баром.

        Args:
            parent: Родительский виджет
        """
        # Статус-бар
        self.status_bar = StatusBar(parent)
        self.status_bar.pack(side="bottom", fill="x")

    def _on_recognition_click(self):
        """Обработчик нажатия кнопки распознавания."""
        print("Кнопка РАСПОЗНАВАНИЕ нажата")
        # TODO: Переход на экран распознавания
        # self.main_window.show_screen("recognition")

    def _on_registration_click(self):
        """Обработчик нажатия кнопки регистрации."""
        print("Кнопка РЕГИСТРАЦИЯ нажата")
        # TODO: Переход на экран регистрации
        # self.main_window.show_screen("registration")

    def _on_database_click(self):
        """Обработчик нажатия кнопки базы данных."""
        print("Кнопка БАЗА ДАННЫХ нажата")
        # TODO: Переход на экран базы данных
        # self.main_window.show_screen("database")

    def _on_settings_click(self):
        """Обработчик нажатия кнопки настроек."""
        print("Кнопка НАСТРОЙКИ нажата")
        # TODO: Переход на экран настроек
        # self.main_window.show_screen("settings")

    def show(self):
        """Переопределяем метод show для обновления статус-бара."""
        super().show()
        # Здесь можно обновить данные статус-бара при показе экрана
        # Например, загрузить актуальное количество записей в БД
