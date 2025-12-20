#!/usr/bin/env python3
"""
Точка входа для GUI приложения системы распознавания лиц.
"""

import logging
import sys

# Настройка логирования ДО всех импортов
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("ЗАПУСК ПРИЛОЖЕНИЯ")
logger.info("=" * 60)

try:
    logger.info("Импортирую MainWindow...")
    from gui.main_window import MainWindow

    logger.info("✓ MainWindow импортирован")

    logger.info("Импортирую MainMenuScreen...")
    from gui.screens.main_menu import MainMenuScreen

    logger.info("✓ MainMenuScreen импортирован")

except Exception as e:
    logger.error(f"✗ ОШИБКА ИМПОРТА: {e}", exc_info=True)
    sys.exit(1)


def main():
    """
    Главная функция приложения.
    Инициализирует главное окно и запускает приложение.
    """
    try:
        # Создаем главное окно
        logger.info("Создаю главное окно...")
        app = MainWindow()
        logger.info("✓ Главное окно создано")

        # Обновляем геометрию окна для корректного расчета размеров виджетов
        logger.info("Обновляю геометрию окна...")
        app.update_idletasks()
        logger.info("✓ Геометрия обновлена")

        # Создаем и добавляем главное меню
        logger.info("Создаю MainMenuScreen...")
        main_menu = MainMenuScreen(app.screen_container, app)
        logger.info("✓ MainMenuScreen создан")

        logger.info("Добавляю экран в app...")
        app.add_screen("main_menu", main_menu)
        logger.info("✓ Экран добавлен")

        # Показываем главное меню
        logger.info("Показываю главное меню...")
        app.show_screen("main_menu")
        logger.info("✓ Главное меню показано")

        logger.info("=" * 60)
        logger.info("ЗАПУСК MAINLOOP")
        logger.info("=" * 60)

        # Запускаем приложение
        app.run()

    except Exception as e:
        logger.error("=" * 60)
        logger.error("КРИТИЧЕСКАЯ ОШИБКА")
        logger.error("=" * 60)
        logger.error(f"Ошибка: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()