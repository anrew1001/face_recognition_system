"""
Цветовая схема приложения Face Recognition.
Основана на архитектуре GUI (темная тема + cyan акценты).
"""

# Основные цвета фона
DARK_BG = "#1A1A2E"           # Основной фон
BG_SECONDARY = "#2D2D2D"      # Вторичный фон
BG_TERTIARY = "#3D3D3D"       # Третичный фон
STATUS_BG = "#0F0F1E"         # Фон статус-бара (темнее основного)

# Акцентные цвета
CYAN_ACCENT = "#00B4D8"       # Основной акцент (кнопки, выделение)
CYAN_HOVER = "#0096B8"        # Hover для cyan кнопок
CYAN_PRESSED = "#007A9A"      # Нажатие на cyan кнопки

# Цвета текста
LIGHT_TEXT = "#FFFFFF"        # Основной текст
SECONDARY_TEXT = "#A0A0A0"    # Вторичный текст (серый)

# Статусные цвета (для bounding boxes и индикаторов)
GREEN_SUCCESS = "#28A745"     # ALIVE + распознан
ORANGE_WARN = "#FD7E14"       # ALIVE + неизвестен
RED_DANGER = "#DC3545"        # NOT ALIVE

# Дополнительные цвета
TRANSPARENT = "transparent"   # Прозрачный фон для кнопок с границей