#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import re
import logging
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
import argparse

# Получаем путь к директории скрипта - будем использовать для создания относительных путей
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Абсолютный путь к корневой директории проекта (где находится api_manager.py)
PROJECT_ROOT = SCRIPT_DIR

# Настройка логирования с относительными путями
log_file_path = os.path.join(".", "patient_bot.log")
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
root_logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
console_handler.setLevel(logging.WARNING)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
logger = logging.getLogger("patient_bot")

# Проверяем аргументы командной строки для уровня логирования
if len(sys.argv) > 2 and sys.argv[2] == '--debug':
    logger.setLevel(logging.DEBUG)
    print("Включен режим отладки с подробным логированием")

# Функция для преобразования путей в безопасный формат для вставки в код
def normalize_path_for_code(path):
    """
    Преобразует путь в безопасный формат для вставки в Python-код
    
    Args:
        path (str): Исходный путь
        
    Returns:
        str: Безопасный путь для вставки в код
    """
    # Нормализуем слеши для текущей ОС
    path = os.path.normpath(path)
    # Заменяем обратные слеши на прямые для совместимости со всеми ОС в Python
    path = path.replace('\\', '/')
    return path

def create_vector_store(case_id):
    """
    Создает векторное хранилище на основе данных о пациенте
    
    Args:
        case_id (str): Идентификатор случая
        
    Returns:
        bool: True в случае успеха, False в случае ошибки
    """
    print("Создание векторного хранилища для данных пациента...")
    logger.debug("Начало создания векторного хранилища для %s", case_id)
    
    # Проверяем наличие файлов данных с использованием относительных путей
    case_dir = os.path.join("cases", case_id)
    instructions_path = os.path.join(case_dir, "bot_instructions.txt")
    factual_data_path = os.path.join(case_dir, "factual_data.txt")
    
    if not os.path.exists(instructions_path) or not os.path.exists(factual_data_path):
        logger.error(f"Отсутствуют необходимые файлы в директории {case_dir}")
        return False
    
    # Загружаем данные
    with open(factual_data_path, 'r', encoding='utf-8') as f:
        factual_data = f.read()
    
    # Создаем директорию для хранилища с относительным путем
    vector_store_path = os.path.join(case_dir, "chroma_db")
    os.makedirs(vector_store_path, exist_ok=True)
    
    try:
        # Инициализируем ChromaDB
        print("Инициализация ChromaDB...")
        chroma_client = chromadb.PersistentClient(path=vector_store_path)
        
        # Создаем или получаем коллекцию
        try:
            collection = chroma_client.get_collection("patient_data")
            print("Используем существующую коллекцию 'patient_data'")
        except:
            # Создаем функцию эмбеддинга на основе многоязычной модели
            print("Создание новой коллекции 'patient_data'...")
            try:
                ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="paraphrase-multilingual-MiniLM-L12-v2"
                )
            except Exception as e:
                logger.error(f"Критическая ошибка: Не удалось создать функцию эмбеддинга SentenceTransformer: {str(e)}")
                print(f"Критическая ошибка: Не удалось создать функцию эмбеддинга: {str(e)}")
                print("Создание векторного хранилища прервано.")
                return False # Завершаем функцию с ошибкой
                
            collection = chroma_client.create_collection(
                name="patient_data",
                embedding_function=ef
            )
        
        # Разбиваем данные на фрагменты для индексации
        print("Обработка и индексация данных пациента...")
        chunks = []
        metadatas = []
        ids = []
        
        # Очищаем коллекцию перед добавлением новых данных
        try:
            # Метод 1: Удаление по IDs
            logger.debug("Попытка очистки коллекции по ID")
            result = collection.get()
            if result and result['ids']:
                logger.debug(f"Получено {len(result['ids'])} ID для удаления")
                print(f"Удаление {len(result['ids'])} существующих документов по ID...")
                collection.delete(ids=result['ids'])
                logger.debug("Успешное удаление документов по ID")
            # Метод 2: Если метод 1 не сработал, попробуем удалить по категории
            else:
                logger.debug("Не найдены ID для удаления, попытка удаления по категории")
                print("Удаление документов по категории...")
                collection.delete(where={"category": {"$ne": "non_existent_category"}})
                logger.debug("Успешное удаление документов по категории")
            
            logger.debug("Проверка количества документов после удаления")
            count = collection.count()
            logger.debug(f"Количество документов в коллекции после удаления: {count}")
        except Exception as e:
            logger.error(f"Ошибка при очистке существующей коллекции ChromaDB: {str(e)}")
            print(f"Ошибка: Не удалось очистить существующую коллекцию ChromaDB: {str(e)}")
            print("Создание векторного хранилища прервано, чтобы избежать дублирования данных.")
            return False # Завершаем функцию с ошибкой
        
        # Разбираем данные пациента на категории
        pattern = r'#{2,3}\s+([^#\n]+)'
        categories = re.findall(pattern, factual_data)
        
        chunks_count = 0
        
        if categories:
            # Разбиваем по категориям
            for i, category in enumerate(categories):
                category = category.strip()
                
                # Определяем начало текущей категории
                if i < len(categories) - 1:
                    start_idx = factual_data.find(f"# {category}")
                    if start_idx == -1:
                        start_idx = factual_data.find(f"## {category}")
                    if start_idx == -1:
                        start_idx = factual_data.find(f"### {category}")
                        
                    end_idx = factual_data.find(f"# {categories[i+1]}")
                    if end_idx == -1:
                        end_idx = factual_data.find(f"## {categories[i+1]}")
                    if end_idx == -1:
                        end_idx = factual_data.find(f"### {categories[i+1]}")
                    
                    category_text = factual_data[start_idx:end_idx].strip()
                else:
                    # Последняя категория
                    start_idx = factual_data.find(f"# {category}")
                    if start_idx == -1:
                        start_idx = factual_data.find(f"## {category}")
                    if start_idx == -1:
                        start_idx = factual_data.find(f"### {category}")
                        
                    category_text = factual_data[start_idx:].strip()
                
                # Удаляем заголовок из текста
                if category_text.startswith(f"# {category}"):
                    category_text = category_text[len(f"# {category}"):].strip()
                elif category_text.startswith(f"## {category}"):
                    category_text = category_text[len(f"## {category}"):].strip()
                elif category_text.startswith(f"### {category}"):
                    category_text = category_text[len(f"### {category}"):].strip()
                
                # Разбиваем текст категории на параграфы
                paragraphs = re.split(r'\n\s*\n', category_text)
                
                for j, paragraph in enumerate(paragraphs):
                    paragraph = paragraph.strip()
                    if len(paragraph) > 10:  # Игнорируем слишком короткие параграфы
                        chunk_id = f"{category.lower().replace(' ', '_')}_{j}"
                        chunks.append(paragraph)
                        metadatas.append({"category": category})
                        ids.append(chunk_id)
                        chunks_count += 1
        else:
            # Если нет категорий, разбиваем на параграфы
            paragraphs = re.split(r'\n\s*\n', factual_data)
            for j, paragraph in enumerate(paragraphs):
                paragraph = paragraph.strip()
                if len(paragraph) > 10:  # Игнорируем слишком короткие параграфы
                    chunk_id = f"paragraph_{j}"
                    chunks.append(paragraph)
                    metadatas.append({"category": "общая информация"})
                    ids.append(chunk_id)
                    chunks_count += 1
        
        # Добавляем чанки в коллекцию
        if chunks:
            logger.debug(f"Добавление {len(chunks)} фрагментов в коллекцию")
            collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            logger.debug(f"Фрагменты успешно добавлены в коллекцию")
            print(f"Добавлено {chunks_count} фрагментов текста в векторное хранилище.")
        else:
            logger.warning("Не удалось извлечь фрагменты текста для индексации.")
            print("Не удалось извлечь фрагменты текста для индексации.")
            
        # Проверяем результат
        count = collection.count()
        print(f"Всего документов в коллекции: {count}")
        
        return True
    except Exception as e:
        logger.error(f"Ошибка при создании векторного хранилища: {str(e)}")
        print(f"Произошла ошибка: {str(e)}")
        return False

def create_patient_bot(case_id):
    """
    Создает бота-пациента, который использует ChromaDB для поиска данных
    
    Args:
        case_id (str): Идентификатор случая
        
    Returns:
        bool: True в случае успеха, False в случае ошибки
    """
    print("Создание бота-пациента с использованием ChromaDB...")
    logger.debug(f"Начало создания бота-пациента для случая {case_id}")
    
    # Получаем абсолютный путь к корневой директории проекта
    project_root = os.path.abspath(PROJECT_ROOT)
    # Преобразуем путь в безопасный формат для вставки в код
    normalized_path = normalize_path_for_code(project_root)
    
    # Проверяем наличие необходимых файлов в корневой директории
    api_manager_exists = os.path.exists(os.path.join(project_root, "api_manager.py"))
    config_exists = os.path.exists(os.path.join(project_root, "config.py"))
    
    if not api_manager_exists or not config_exists:
        logger.warning(f"Внимание: Не найдены необходимые файлы в директории проекта: {project_root}")
        logger.warning(f"api_manager.py существует: {api_manager_exists}")
        logger.warning(f"config.py существует: {config_exists}")
        print(f"Внимание: Не все необходимые файлы найдены в директории проекта.")
        print(f"Путь к проекту: {project_root}")
    else:
        print(f"Обнаружены необходимые файлы для импорта в директории проекта: {project_root}")
    
    # Проверяем наличие векторного хранилища с относительными путями
    case_dir = os.path.join("cases", case_id)
    vector_store_path = os.path.join(case_dir, "chroma_db")
    
    if not os.path.exists(vector_store_path):
        # Если хранилища нет, создаем его
        logger.debug(f"Векторное хранилище не найдено, создаем новое: {vector_store_path}")
        success = create_vector_store(case_id)
        if not success:
            logger.error("Не удалось создать векторное хранилище")
            return False
    else:
        logger.debug(f"Найдено существующее векторное хранилище: {vector_store_path}")
    
    # Загружаем инструкции и информацию о пациенте
    instructions_path = os.path.join(case_dir, "bot_instructions.txt")
    factual_data_path = os.path.join(case_dir, "factual_data.txt")
    
    if not os.path.exists(instructions_path) or not os.path.exists(factual_data_path):
        logger.error(f"Отсутствуют необходимые файлы в директории {case_dir}")
        return False
    
    # Загружаем данные
    with open(instructions_path, 'r', encoding='utf-8') as f:
        instructions = f.read()
        print(f"Загружены инструкции бота ({len(instructions)} байт)")
    
    # Создаем директорию для бота
    bot_dir = os.path.join(case_dir, "bot")
    os.makedirs(bot_dir, exist_ok=True)
    
    try:
        # Разбиваем генерацию кода бота на статические и динамические части
        # Часть 1: Импорты и функция поиска проекта (статическая)
        bot_code_part1 = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Бот-пациент с ChromaDB

import os
import sys
import json
import re
import logging
import chromadb
from datetime import datetime

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
    return None'''
        
        # Часть 2: Путь к проекту (динамическая)
        bot_code_part2 = f'''

# Абсолютный путь к корневой директории проекта, зафиксированный при генерации бота
FIXED_PROJECT_ROOT = "{normalized_path}"

# Выбор пути к проекту
if os.path.exists(FIXED_PROJECT_ROOT) and os.path.exists(os.path.join(FIXED_PROJECT_ROOT, "api_manager.py")):
    # Используем зафиксированный путь
    PROJECT_ROOT = FIXED_PROJECT_ROOT
    print(f"Используем зафиксированный путь к проекту: {{PROJECT_ROOT}}")
else:
    # Динамически ищем директорию проекта
    PROJECT_ROOT = find_project_root()
    if PROJECT_ROOT:
        print(f"Динамически обнаружен путь к проекту: {{PROJECT_ROOT}}")
    else:
        print("Не удалось определить директорию проекта!")
        print("Текущая директория:", os.path.dirname(os.path.abspath(__file__)))
        print("Путь к проекту должен содержать файл api_manager.py")
        sys.exit(1)

# Добавляем путь к проекту в sys.path для импорта модулей
sys.path.insert(0, PROJECT_ROOT)'''
        
        # Часть 3: Импорты и настройка логирования (статическая)
        bot_code_part3 = '''

# Импортируем api_manager и конфигурацию из основного проекта
try:
    from api_manager import api_manager
    from config import MODELS
    print("Успешно импортированы api_manager и config")
except ImportError as e:
    print(f"Ошибка импорта модулей: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

# === Настройка логирования: только WARNING и выше в консоль, всё в файл ===
log_file_path = os.path.join(".", "patient_bot.log")
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
root_logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
console_handler.setLevel(logging.WARNING)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
logger = logging.getLogger("patient_bot")'''
        
        # Часть 4: Начало класса PatientBot (статическая)
        bot_code_part4 = '''

class PatientBot:
    """Класс бота-пациента с ChromaDB"""
    
    def __init__(self):
        """Инициализация бота"""
        # Загружаем инструкции для бота'''
        
        # Часть 5: Инструкции бота (динамическая)
        bot_code_part5 = f'''
        self.instructions = """{instructions}"""'''
        
        # Часть 6: Инициализация ChromaDB (статическая)
        bot_code_part6 = '''
        
        # Получаем путь к текущей директории для относительных путей
        self.current_dir = "."
        
        # Используем api_manager из основного проекта
        self.api_manager = api_manager
        
        # Инициализируем ChromaDB с относительным путем
        # Получаем абсолютный путь к текущему файлу скрипта
        script_path = os.path.abspath(__file__)
        # Получаем директорию бота
        bot_dir = os.path.dirname(script_path)
        # Получаем родительскую директорию (директорию кейса)
        case_dir = os.path.dirname(bot_dir)
        # Определяем путь к chroma_db в директории кейса
        self.chroma_dir = os.path.join(case_dir, "chroma_db")
        if not os.path.exists(self.chroma_dir):
            logger.error(f"Директория ChromaDB не найдена: {self.chroma_dir}")
            sys.exit(1)
        
        # Подключаемся к векторному хранилищу
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
            self.collection = self.chroma_client.get_collection("patient_data")
            logger.info(f"Успешное подключение к ChromaDB")
        except Exception as e:
            logger.error(f"Ошибка подключения к ChromaDB: {str(e)}")
            sys.exit(1)
        
        # Сохраняем историю сообщений
        self.messages = []
    
    def _search_in_chroma(self, query, n_results=5):
        """
        Поиск в векторном хранилище
        
        Args:
            query (str): Текст запроса
            n_results (int): Количество результатов
            
        Returns:
            list: Найденные документы
        """
        try:
            # Выполняем поиск в ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Обрабатываем результаты
            documents = []
            if results and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {"category": "unknown"}
                    documents.append({
                        "text": doc,
                        "category": metadata["category"]
                    })
            
            return documents
        except Exception as e:
            logger.error(f"Ошибка при поиске в ChromaDB: {str(e)}")
            return []
    
    def _generate_response(self, user_message):
        """
        Генерирует ответ на сообщение пользователя
        
        Args:
            user_message (str): Сообщение пользователя
            
        Returns:
            str: Ответ бота
        """
        # Поиск релевантной информации в векторном хранилище
        relevant_docs = self._search_in_chroma(user_message)
        relevant_info = ""
        
        if relevant_docs:
            relevant_info = "\\n\\n".join([f"ИНФОРМАЦИЯ ({doc['category']}): {doc['text']}" for doc in relevant_docs])
        
        # Формируем историю сообщений для модели
        messages = [
            {"role": "system", "content": f"{self.instructions}\\n\\nНАЙДЕННАЯ ИНФОРМАЦИЯ О ПАЦИЕНТЕ:\\n{relevant_info}"}
        ]
        
        # Добавляем историю диалога
        for msg in self.messages:
            messages.append(msg)
        
        # Добавляем текущий вопрос
        messages.append({"role": "user", "content": user_message})
        
        try:
            # Используем api_manager для перебора всех комбинаций моделей и ключей
            def make_request(model, client):
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                )
            
            # Используем execute_with_retry для перебора всех возможных комбинаций
            response = self.api_manager.execute_with_retry("dialogue", make_request)
            
            # Проверяем, что получили ответ
            if not response:
                raise Exception("Не удалось получить ответ от API после перебора всех комбинаций")
            
            # Получаем ответ
            response_text = response.choices[0].message.content.strip()
            
            # Проверяем, что ответ от лица пациента
            if response_text.startswith("Врач:") or response_text.startswith("Доктор:"):
                response_text = "Извините, я не понял ваш вопрос. Вы можете уточнить?"
            
            return response_text
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {str(e)}")
            return "Извините, произошла ошибка. Не могу ответить на ваш вопрос сейчас."
    
    def start_conversation(self):
        """Запускает диалог с ботом"""
        print("=== Бот-пациент запущен ===")
        print("Вы - врач, общающийся с пациентом. Задавайте вопросы и проведите обследование.")
        print("Для выхода введите '/выход' или 'exit'")
        print()
        
        # Автоматически отправляем первое сообщение от врача
        first_message = "Здравствуйте! Что Вас беспокоит? На что хотите пожаловаться?"
        print(f"Врач: {first_message}")
        
        # Сохраняем первое сообщение в истории
        self.messages.append({"role": "user", "content": first_message})
        
        # Генерируем ответ пациента на первое сообщение
        response = self._generate_response(first_message)
        
        # Выводим ответ и сохраняем в истории
        print(f"Пациент: {response}")
        self.messages.append({"role": "assistant", "content": response})
        
        while True:
            user_input = input("Врач: ")
            
            if user_input.lower() in ['/выход', 'exit', 'quit']:
                print("Диалог завершен.")
                break
            
            # Сохраняем сообщение пользователя
            self.messages.append({"role": "user", "content": user_input})
            
            # Генерируем ответ
            response = self._generate_response(user_input)
            
            # Выводим ответ и сохраняем в истории
            print(f"Пациент: {response}")
            self.messages.append({"role": "assistant", "content": response})

def main():
    """Основная функция"""
    bot = PatientBot()
    bot.start_conversation()

if __name__ == "__main__":
    main()'''

        # Соединяем все части кода
        bot_code = bot_code_part1 + bot_code_part2 + bot_code_part3 + bot_code_part4 + bot_code_part5 + bot_code_part6
        
        # Сохраняем файл бота
        bot_file_path = os.path.join(bot_dir, "vector_patient_bot.py")
        with open(bot_file_path, 'w', encoding='utf-8') as f:
            f.write(bot_code)
            
        print(f"Файл бота создан успешно: {bot_file_path}")
        
        # Создаем файл запуска с разделением на части
        # Часть 1: Импорты и функция поиска проекта (статическая)
        run_file_part1 = '''#!/usr/bin/env python3
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
    return None'''
        
        # Часть 2: Путь к проекту (динамическая)
        run_file_part2 = f'''

# Абсолютный путь к корневой директории проекта, зафиксированный при генерации бота
FIXED_PROJECT_ROOT = "{normalized_path}"

# Выбор пути к проекту
if os.path.exists(FIXED_PROJECT_ROOT) and os.path.exists(os.path.join(FIXED_PROJECT_ROOT, "api_manager.py")):
    # Используем зафиксированный путь
    PROJECT_ROOT = FIXED_PROJECT_ROOT
    print(f"Используем зафиксированный путь к проекту: {{PROJECT_ROOT}}")
else:
    # Динамически ищем директорию проекта
    PROJECT_ROOT = find_project_root()
    if PROJECT_ROOT:
        print(f"Динамически обнаружен путь к проекту: {{PROJECT_ROOT}}")
    else:
        print("Не удалось определить директорию проекта!")
        print("Текущая директория:", os.path.dirname(os.path.abspath(__file__)))
        print("Путь к проекту должен содержать файл api_manager.py")
        sys.exit(1)

# Добавляем путь к проекту в sys.path для импорта модулей
sys.path.insert(0, PROJECT_ROOT)'''
        
        # Часть 3: Импорты и основная функция (статическая)
        run_file_part3 = '''

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
    main()'''
        
        # Соединяем все части кода
        run_file_code = run_file_part1 + run_file_part2 + run_file_part3
        
        run_file_path = os.path.join(bot_dir, "run_vector_bot.py")
        with open(run_file_path, 'w', encoding='utf-8') as f:
            f.write(run_file_code)
            
        print(f"Файл запуска создан успешно: {run_file_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при создании бота-пациента: {str(e)}")
        print(f"Произошла ошибка: {str(e)}")
        return False

def main():
    """Основная функция"""
    print("=== Создание бота-пациента с использованием ChromaDB ===")
    
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1:
        case_id = sys.argv[1]
    else:
        # По умолчанию используем существующий случай
        case_id = "case_20250320_220320"
    
    print(f"Создание бота-пациента для случая: {case_id}")
    success = create_patient_bot(case_id)
    
    if success:
        print("\nБот-пациент успешно создан с ChromaDB!")
        
        # Формируем путь к скрипту запуска с относительным путем
        run_script_path = os.path.join("cases", case_id, "bot", "run_vector_bot.py")
        
        # Выводим команду запуска
        print(f"Запустите: python \"{run_script_path}\"")
    else:
        print("\nСоздание бота не удалось.")
        sys.exit(1)

if __name__ == "__main__":
    main()
