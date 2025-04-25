#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Скрипт запуска бота-пациента с ChromaDB

import os
import sys

# Функция для поиска корневой директории проекта
def find_project_root():
    """Определяет корневую директорию проекта по наличию ключевых файлов"""
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):  # Ограничиваем глубину поиска
        if os.path.exists(os.path.join(current, "api_manager.py")):
            return current
        parent = os.path.dirname(current)
        if parent == current:  # Достигли корня файловой системы
            return None
        current = parent
    return None

# Абсолютный путь к корневой директории проекта, зафиксированный при генерации бота
FIXED_PROJECT_ROOT = "/mnt/storage/BOTS/AIpatient"

# Выбор пути к проекту
if os.path.exists(FIXED_PROJECT_ROOT) and os.path.exists(os.path.join(FIXED_PROJECT_ROOT, "api_manager.py")):
    # Используем зафиксированный путь
    PROJECT_ROOT = FIXED_PROJECT_ROOT
    print(f"Используем зафиксированный путь к проекту: {PROJECT_ROOT}")
else:
    # Динамически ищем директорию проекта
    PROJECT_ROOT = find_project_root()
    if PROJECT_ROOT:
        print(f"Динамически обнаружен путь к проекту: {PROJECT_ROOT}")
    else:
        print("Не удалось определить директорию проекта!")
        print("Текущая директория:", os.path.dirname(os.path.abspath(__file__)))
        print("Путь к проекту должен содержать файл api_manager.py")
        sys.exit(1)

# Добавляем путь к проекту в sys.path для импорта модулей
sys.path.insert(0, PROJECT_ROOT)

# Проверяем доступность модулей
try:
    # Сначала проверяем импорт api_manager и config
    from api_manager import api_manager
    from config import MODELS
    print("api_manager и config успешно импортированы")
    
    # Импортируем PatientBot
    from vector_patient_bot import PatientBot
    print("PatientBot успешно импортирован")
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

def main():
    print(f"=== Запуск бота-пациента с ChromaDB ===")
    bot = PatientBot()
    bot.start_conversation()

if __name__ == "__main__":
    main()