"""
Конфигурационный файл для хранения API ключей и настроек моделей.
"""

import os
from dotenv import load_dotenv
import logging # Добавим для предупреждения

# Загружает переменные из файла .env в окружение
load_dotenv()

# Получаем строку ключей из переменной окружения
keys_string = os.getenv("GOOGLE_API_KEYS")

# Разделяем строку по запятым и удаляем возможные пробелы вокруг ключей
if keys_string:
    GOOGLE_API_KEYS = [key.strip() for key in keys_string.split(',')]
else:
    GOOGLE_API_KEYS = []

# Предупреждение, если ключи не найдены
if not GOOGLE_API_KEYS:
    # Используем logging, если он настроен, или print
    try:
        logging.warning("Переменная окружения GOOGLE_API_KEYS не найдена или пуста. API запросы могут не работать.")
    except NameError: # Если logging еще не настроен на момент импорта config
         print("ПРЕДУПРЕЖДЕНИЕ: Переменная окружения GOOGLE_API_KEYS не найдена или пуста.")
else:
    logging.info(f"Загружено {len(GOOGLE_API_KEYS)} API ключей из переменных окружения.") # Сообщение об успехе

# Настройки моделей для разных компонентов системы
MODELS = {
    # Модели для диалога куратора (основная и запасные)
    "dialogue": [
        "emini-1.5-flash",
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-pro-preview-03-25",
        "gemini-2.0-flash-thinking-exp-1219",
        "gemini-2.0-flash",
    ],
    
    # Модели для синтеза информации (основная и запасные)
    "synthesis": [
        "gemini-1.5-flash",
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-pro-preview-03-25",
        "gemini-2.0-flash-thinking-exp-1219",
        "gemini-2.0-flash",
    ],
    
    # Модели для арбитража (основная и запасные)
    "arbitrator": [
        "gemini-1.5-flash",
        "gemini-2.5-pro-preview-03-25",
        "gemini-2.0-flash-thinking-exp-1219",
        "gemini-2.0-flash",
    ]
}

# Таймаут между попытками запроса (в секундах)
API_RETRY_TIMEOUT = 2 

# Максимальное количество повторных попыток запроса при ошибках API
MAX_API_RETRIES = 3 