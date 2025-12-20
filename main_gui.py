#!/usr/bin/env python3
"""
Точка входа для GUI приложения системы распознавания лиц.
"""

from gui.main_window import MainWindow
from gui.screens.main_menu import MainMenuScreen


def main():
    """
    Главная функция приложения.
    Инициализирует главное окно и запускает приложение.
    """
    # Создаем главное окно
    app = MainWindow()

    # Обновляем геометрию окна для корректного расчета размеров виджетов
    app.update_idletasks()

    # Создаем и добавляем главное меню
    main_menu = MainMenuScreen(app.screen_container, app)
    app.add_screen("main_menu", main_menu)

    # Показываем главное меню
    app.show_screen("main_menu")

    # Запускаем приложение
    app.run()


if __name__ == "__main__":
    main()
