import os
import sys
import json
import logging
import random
import argparse
from datetime import datetime
from openai import OpenAI
from api_manager import api_manager
from config import MODELS
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

# Получаем путь к директории скрипта для создания относительных путей
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Добавляем директорию скрипта в sys.path для корректного импорта
# (это не нужно менять, т.к. это не влияет на файловую систему)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Добавляю загрузку инструкций из внешнего файла
def load_bot_instructions(instruction_type):
    """
    Загружает инструкции для ботов из файла bot_instructions.py
    
    Args:
        instruction_type: Тип инструкции (system_prompt, factual_data_arbitrator, instructions_arbitrator, extraction_prompt)
        
    Returns:
        str: Текст инструкции
    """
    try:
        # Импортируем инструкции из файла в той же директории
        import bot_instructions
        
        # Получаем соответствующую инструкцию
        if instruction_type == "system_prompt":
            return bot_instructions.SYSTEM_PROMPT
        elif instruction_type == "factual_data_arbitrator":
            return bot_instructions.FACTUAL_DATA_ARBITRATOR_PROMPT
        elif instruction_type == "instructions_arbitrator":
            return bot_instructions.INSTRUCTIONS_ARBITRATOR_PROMPT
        elif instruction_type == "instruction_extraction_prompt":
            return bot_instructions.INSTRUCTION_EXTRACTION_PROMPT
        elif instruction_type == "factual_data_extraction_prompt":
            return bot_instructions.FACTUAL_DATA_EXTRACTION_PROMPT
        else:
            raise ValueError(f"Неизвестный тип инструкции: {instruction_type}")
    except ImportError:
        # Если файл с инструкциями не найден, выводим ошибку и завершаем программу
        print("Ошибка: Файл bot_instructions.py не найден в директории программы")
        logging.error("Ошибка: Файл bot_instructions.py не найден в директории программы")
        sys.exit(1)
    except AttributeError as e:
        # Если в файле нет нужной инструкции
        print(f"Ошибка: В файле bot_instructions.py не найдена инструкция {instruction_type}: {str(e)}")
        logging.error(f"Ошибка: В файле bot_instructions.py не найдена инструкция {instruction_type}: {str(e)}")
        sys.exit(1)

# Добавляю импорт функции создания бота из patient_bot_generator
from patient_bot_generator import create_patient_bot

# === ДОБАВИТЬ ФУНКЦИЮ ДЛЯ ГЛОБАЛЬНОГО ЛОГИРОВАНИЯ ===
def setup_global_logging(debug_mode):
    """
    Централизованная настройка root logger для управления выводом в консоль.
    В debug-режиме выводит все, иначе только WARNING и выше.
    """
    root_logger = logging.getLogger()
    # Удаляем все существующие хендлеры
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    if debug_mode:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.WARNING)
    root_logger.addHandler(console_handler)

class DocumentArbitrator:
    def __init__(self, debug_mode=False):
        """
        Инициализация арбитра для проверки документов
        
        Args:
            debug_mode (bool): Режим отладки с потоковым выводом ответов моделей
        """
        self.logger = logging.getLogger("arbitrator")
        
        # Настройка логирования
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Сохраняем режим отладки
        self.debug_mode = debug_mode
        
    def arbitrate_and_improve_document(self, dialogue_history, document, document_type):
        """
        Выполняет оценку и улучшение документа с помощью LLM.
        Всегда возвращает результат обработки LLM как улучшенную версию.

        Args:
            dialogue_history: Список сообщений, представляющих диалог.
            document: Текст исходного сгенерированного документа.
            document_type: "factual_data" или "instructions".

        Returns:
            str: Текст документа, обработанный арбитром, или исходный документ в случае ошибки API.
        """
        # Используем всю переданную историю диалога без обрезания,
        # так как теперь мы получаем уже чистую историю без служебных сообщений
        truncated_history = dialogue_history

        # Выбор системного промпта (оставляем как есть)
        if document_type == "factual_data":
            # Убедитесь, что промпт FACTUAL_DATA_ARBITRATOR_PROMPT существует и загружается
            system_prompt = self._get_factual_data_arbitrator_prompt()
        else:  # instructions
            system_prompt = self._get_instructions_arbitrator_prompt()

        # Формирование сообщения для LLM (оставляем как есть, можно улучшить заголовок)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._format_dialogue(truncated_history) +
                # Можно немного изменить текст, чтобы было понятнее, что это вход для улучшения
                "\n\n### Исходный сгенерированный документ для улучшения ###\n\n" + document}
        ]

        def make_request(model, client):
            return client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True  # Добавляем потоковый режим
            )

        try:
            self.logger.info(f"Отправка запроса арбитру для улучшения документа типа: {document_type}")

            # Используем менеджер API (оставляем как есть)
            stream_completion = api_manager.execute_with_retry("arbitrator", make_request)

            if not stream_completion:
                raise Exception("Не удалось получить ответ от API после нескольких попыток")

            # Обрабатываем потоковый ответ
            final_document = ""
            
            # Если включен режим отладки, выводим потоковый заголовок
            if self.debug_mode:
                print(f"\nАрбитраж {document_type} (потоково): ", end="", flush=True)
            
            # Итерируем по потоку частей ответа
            for chunk in stream_completion:
                if not hasattr(chunk, 'choices') or not chunk.choices:
                    continue
                
                delta_content = chunk.choices[0].delta.content
                if delta_content is not None:
                    # Если включен режим отладки, выводим каждую часть
                    if self.debug_mode:
                        print(delta_content, end="", flush=True)
                    
                    # Собираем полный ответ из частей
                    final_document += delta_content
            
            # Если включен режим отладки, добавляем перевод строки после потока
            if self.debug_mode:
                print()
            
            self.logger.info(f"Получен обработанный документ ({document_type}) от арбитра.")

            # Просто возвращаем результат работы LLM
            # Теперь метод возвращает только одну строку - текст документа
            return final_document.strip()

        except Exception as e:
            error_msg = f"Ошибка при арбитраже документа ({document_type}): {str(e)}"
            self.logger.error(error_msg)
            # В случае ошибки API возвращаем исходный документ, чтобы не потерять данные
            print(f"\nПРЕДУПРЕЖДЕНИЕ: Ошибка при обработке документа арбитром ({document_type}). Используется исходная версия.")
            return document
            
    def _get_factual_data_arbitrator_prompt(self):
        """Возвращает системный промпт для арбитра фактических данных"""
        return load_bot_instructions("factual_data_arbitrator")
    
    def _get_instructions_arbitrator_prompt(self):
        """Возвращает системный промпт для арбитра инструкций"""
        return load_bot_instructions("instructions_arbitrator")
    
    def _format_dialogue(self, messages):
        """Форматирует историю диалога в читаемый текст"""
        formatted = "### История диалога ###\n\n"
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                continue  # Пропускаем системные сообщения в выводе
            elif role == "user":
                formatted += f"Преподаватель: {content}\n\n"
            elif role == "assistant":
                formatted += f"Бот-куратор: {content}\n\n"
                
        return formatted

class CuratorAgent:
    def __init__(self, debug_mode=False):
        # Настройка логирования
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._setup_logging()
        
        # История сообщений для контекста
        self.messages = [
            {"role": "system", "content": self._get_system_prompt()}
        ]
        
        # Структура для хранения данных о пациенте
        self.patient_data = {
            "demographics": {"age": None, "gender": None},
            "complaints": {},
            "history": {"disease": None, "life": None},
            "examination": {},
            "tests": {},
            "diagnosis": None,
            "treatment": [],
            "teaching_algorithm": {
                "stages": [],
                "evaluation_criteria": {}
            }
        }
        
        # Флаг завершения сбора данных
        self.data_collection_complete = False
        
        # Режим отладки
        self.debug_mode = debug_mode
        
        self.first_prompt = True
        
    def _setup_logging(self):
        """Настраивает логирование для текущей сессии"""
        logs_dir = os.path.join("logs", self.session_id)
        os.makedirs(logs_dir, exist_ok=True)
        
        self.logger = logging.getLogger(self.session_id)
        self.logger.setLevel(logging.INFO)
        
        # Форматирование
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Файловый хендлер
        file_handler = logging.FileHandler(os.path.join(logs_dir, "dialogue.log"), encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # === УДАЛЯЕМ КОНСОЛЬНЫЙ ХЕНДЛЕР ===
        # Консольный вывод теперь глобально управляется через setup_global_logging
        self.logger.info(f"Начата новая сессия: {self.session_id}")
    
    def _get_system_prompt(self):
        """Возвращает системный промпт для агента-куратора"""
        return load_bot_instructions("system_prompt")
    
    def start_conversation(self):
        """Начинает беседу с преподавателем"""
        print("=== Система создания виртуальных пациентов ===")
        print("Агент-куратор поможет вам создать клинический случай.")
        print("Введите начальное описание случая или нажмите Enter для начала диалога с чистого листа.")
        print("Для завершения работы введите 'выход'.")
        print("Для генерации документов и бота введите '/генерация' в любой момент.")
        print()
        if self.first_prompt:
            print("ПОДСКАЗКА: Enter — отправить сообщение")
            self.first_prompt = False
        self.logger.info("Начало диалога")
        
        # Получаем начальный ввод
        initial_input = multiline_input("Описание случая: ")
        self.logger.info(f"Пользователь: {initial_input}")
        
        if initial_input.lower() in ["выход", "exit", "quit", "q"]:
            print("Работа завершена.")
            self.logger.info("Пользователь завершил работу")
            return
        
        if initial_input:
            self.messages.append({"role": "user", "content": initial_input})
            self._process_response()
        
        # Основной цикл диалога
        while not self.data_collection_complete:
            user_input = multiline_input("\nВаш ответ: ")
            self.logger.info(f"Пользователь: {user_input}")
            
            if user_input.lower() in ["выход", "exit", "quit", "q"]:
                print("Работа завершена.")
                self.logger.info("Пользователь завершил работу")
                break
            
            # Проверка на команду генерации
            if user_input.lower() in ["/генерация", "/generate", "/финализация", "/finalize"]:
                self.logger.info("Получена команда генерации документов и бота")
                self._finalize_data_collection()
                break
            
            self.messages.append({"role": "user", "content": user_input})
            self._process_response()
            
            # Проверяем, завершен ли сбор данных
            if "завершить" in user_input.lower() or "готово" in user_input.lower():
                self._finalize_data_collection()
    
    def _process_response(self):
        """Обрабатывает ответ от преподавателя и генерирует ответ бота"""
        def make_request(model, client):
            return client.chat.completions.create(
                model=model,
                messages=self.messages,
                stream=True  # Добавляем потоковый режим
            )
        
        try:
            self.logger.info("Отправка запроса к модели диалога")
            
            # Используем менеджер API для выполнения запроса с автоматическим переключением моделей при ошибке
            stream_completion = api_manager.execute_with_retry("dialogue", make_request)
            
            # Добавляем дополнительную проверку, чтобы точно убедиться, что stream_completion не None
            if stream_completion is None:
                raise Exception("Не удалось получить ответ от API после нескольких попыток")
            
            # Обрабатываем потоковый ответ
            full_response = ""
            
            # Если включен режим отладки, выводим потоковый заголовок
            if self.debug_mode:
                print("\nКуратор (потоково): ", end="", flush=True)
            
            # Итерируем по потоку частей ответа
            for chunk in stream_completion:
                if not hasattr(chunk, 'choices') or not chunk.choices:
                    continue
                
                delta_content = chunk.choices[0].delta.content
                if delta_content is not None:
                    # Если включен режим отладки, выводим каждую часть
                    if self.debug_mode:
                        print(delta_content, end="", flush=True)
                    
                    # Собираем полный ответ из частей
                    full_response += delta_content
            
            # Если включен режим отладки, добавляем перевод строки после потока
            if self.debug_mode:
                print()
            # === ОСТАВЛЯЕМ ТОЛЬКО ПОЛЬЗОВАТЕЛЬСКИЙ ВЫВОД ===
            print(f"\nКуратор: {full_response}")
            
            # Добавляем в историю сообщений и логируем
            self.messages.append({"role": "assistant", "content": full_response})
            self.logger.info(f"Куратор: {full_response}")
            
            # Пытаемся обновить данные на основе ответа
            self._update_patient_data(self.messages)
            
        except Exception as e:
            error_msg = f"Ошибка при обработке запроса: {str(e)}"
            self.logger.error(error_msg)
            print(f"\nОшибка: {str(e)}")
    
    def _update_patient_data(self, messages):
        """
        Обновляет структуру данных пациента на основе диалога
        Этот метод может быть улучшен с использованием промпта для извлечения структурированных данных
        """
        # Здесь можно добавить логику для обновления self.patient_data на основе диалога
        # В простом варианте это можно делать при финализации
        pass
    
    def _finalize_data_collection(self):
        """Завершает сбор данных, формирует выходные файлы и запускает арбитраж"""
        print("\nЗавершение сбора данных и формирование файлов...")
        self.logger.info("Начало финализации данных")
        
        # Создаем директорию для сохранения файлов с генерацией случайного ID
        case_id = generate_case_id()
        case_dir = os.path.join("cases", case_id)
        os.makedirs(case_dir, exist_ok=True)
        self.logger.info(f"Создана директория для кейса: {case_dir}")
        
        # Сохраняем копию истории диалога перед генерацией
        # Это позволяет нам сохранить чистый диалог без служебных сообщений
        # и использовать его для генерации инструкций и фактических данных
        clean_dialogue_history = self.messages.copy()
        
        # Сохраняем чистую историю диалога без служебных сообщений
        dialogue_path = os.path.join(case_dir, "dialogue.json")
        with open(dialogue_path, "w", encoding="utf-8") as f:
            json.dump(clean_dialogue_history, f, ensure_ascii=False, indent=2)
        
        # Загружаем отдельные промпты для инструкций и фактических данных
        instruction_prompt = load_bot_instructions("instruction_extraction_prompt")
        factual_data_prompt = load_bot_instructions("factual_data_extraction_prompt")
        
        # Шаг 1: Генерация инструкции для бота-пациента
        print("\nГенерация инструкции для бота-пациента...")
        self.logger.info("Отправка запроса для генерации инструкции для бота-пациента")
        
        # Создаем копию чистой истории и добавляем промпт для инструкций
        instruction_messages = clean_dialogue_history.copy()
        instruction_messages.append({"role": "user", "content": instruction_prompt})
        
        def make_instruction_request(model, client):
            return client.chat.completions.create(
                model=model,
                messages=instruction_messages,
                stream=True  # Добавляем потоковый режим
            )
        
        try:
            # Используем менеджер API для выполнения запроса с автоматическим переключением моделей при ошибке
            stream_completion = api_manager.execute_with_retry("synthesis", make_instruction_request)
            
            if not stream_completion:
                raise Exception("Не удалось получить ответ от API после нескольких попыток")
            
            # Обрабатываем потоковый ответ
            bot_instructions = ""
            
            # Если включен режим отладки, выводим потоковый заголовок
            if self.debug_mode:
                print("\nГенерация инструкций (потоково): ", end="", flush=True)
            
            # Итерируем по потоку частей ответа
            for chunk in stream_completion:
                if not hasattr(chunk, 'choices') or not chunk.choices:
                    continue
                
                delta_content = chunk.choices[0].delta.content
                if delta_content is not None:
                    # Если включен режим отладки, выводим каждую часть
                    if self.debug_mode:
                        print(delta_content, end="", flush=True)
                    
                    # Собираем полный ответ из частей
                    bot_instructions += delta_content
            
            # Если включен режим отладки, добавляем перевод строки после потока
            if self.debug_mode:
                print()
            
            self.logger.info("Получены инструкции для бота-пациента")
            
            # НЕ добавляем в историю диалога, только сохраняем в файл
            
            # Сохраняем исходные инструкции
            bot_instructions_initial_path = os.path.join(case_dir, "bot_instructions_initial.txt")
            with open(bot_instructions_initial_path, "w", encoding="utf-8") as f:
                f.write(bot_instructions)
            self.logger.info(f"Сохранены исходные инструкции для бота: {bot_instructions_initial_path}")
            
            # Инициализируем арбитра
            arbitrator = DocumentArbitrator(self.debug_mode)
            
            # Процесс арбитража для инструкций - используем чистую историю
            print("\nАрбитр обрабатывает инструкции для бота...")
            final_bot_instructions = arbitrator.arbitrate_and_improve_document(
                clean_dialogue_history, bot_instructions, "instructions"
            )
            
            # Всегда считаем, что результат арбитра - это то, что нам нужно
            bot_instructions = final_bot_instructions
            print("Инструкции для бота обработаны арбитром.")
            
            # Сохраняем финальные инструкции
            bot_instructions_path = os.path.join(case_dir, "bot_instructions.txt")
            with open(bot_instructions_path, "w", encoding="utf-8") as f:
                f.write(bot_instructions)
            print(f"- Финальные инструкции для бота сохранены в: bot_instructions.txt")
            
            # Шаг 2: Генерация фактических данных о пациенте
            print("\nГенерация фактических данных о пациенте...")
            self.logger.info("Отправка запроса для генерации фактических данных")
            
            # Создаем копию чистой истории и добавляем промпт для фактических данных
            factual_data_messages = clean_dialogue_history.copy()
            factual_data_messages.append({"role": "user", "content": factual_data_prompt})
            
            def make_factual_data_request(model, client):
                return client.chat.completions.create(
                    model=model,
                    messages=factual_data_messages,
                    stream=True  # Добавляем потоковый режим
                )
            
            # Используем менеджер API для выполнения запроса с автоматическим переключением моделей при ошибке
            stream_completion = api_manager.execute_with_retry("synthesis", make_factual_data_request)
            
            if not stream_completion:
                raise Exception("Не удалось получить ответ от API после нескольких попыток")
            
            # Обрабатываем потоковый ответ
            factual_data = ""
            
            # Если включен режим отладки, выводим потоковый заголовок
            if self.debug_mode:
                print("\nГенерация фактических данных (потоково): ", end="", flush=True)
            
            # Итерируем по потоку частей ответа
            for chunk in stream_completion:
                if not hasattr(chunk, 'choices') or not chunk.choices:
                    continue
                
                delta_content = chunk.choices[0].delta.content
                if delta_content is not None:
                    # Если включен режим отладки, выводим каждую часть
                    if self.debug_mode:
                        print(delta_content, end="", flush=True)
                    
                    # Собираем полный ответ из частей
                    factual_data += delta_content
            
            # Если включен режим отладки, добавляем перевод строки после потока
            if self.debug_mode:
                print()
            
            self.logger.info("Получены фактические данные о пациенте")
            
            # НЕ добавляем в историю диалога, только сохраняем в файл
            
            # Сохраняем исходные фактические данные
            factual_data_initial_path = os.path.join(case_dir, "factual_data_initial.txt")
            with open(factual_data_initial_path, "w", encoding="utf-8") as f:
                f.write(factual_data)
            self.logger.info(f"Сохранены исходные фактические данные: {factual_data_initial_path}")
            
            # Процесс арбитража для фактических данных - используем чистую историю
            print("\nАрбитр обрабатывает фактические данные...")
            final_factual_data = arbitrator.arbitrate_and_improve_document(
                clean_dialogue_history, factual_data, "factual_data"
            )
            
            # Всегда считаем, что результат арбитра - это то, что нам нужно
            factual_data = final_factual_data
            print("Фактические данные обработаны арбитром.")
            
            # Сохраняем финальные фактические данные
            factual_data_path = os.path.join(case_dir, "factual_data.txt")
            with open(factual_data_path, "w", encoding="utf-8") as f:
                f.write(factual_data)
            print(f"- Финальные фактические данные сохранены в: factual_data.txt")
            
            print(f"\nДанные успешно сохранены в директории cases/{case_id}")
            print(f"- История диалога: dialogue.json")
            print(f"- Инструкции для бота: bot_instructions.txt")
            print(f"- Фактические данные: factual_data.txt")
            
            self.logger.info("Финализация данных успешно завершена")
            self.data_collection_complete = True
            
            # Запускаем создание виртуального пациента с векторным хранилищем
            print("\nСоздание виртуального пациента с векторным хранилищем...")
            self.logger.info(f"Запуск создания виртуального пациента для случая {case_id}")
            
            try:
                success = create_patient_bot(case_id)
                if success:
                    print("\nВиртуальный пациент успешно создан!")
                    print(f"Запустите: python {os.path.join('cases', case_id, 'bot', 'run_vector_bot.py')}")
                    self.logger.info(f"Виртуальный пациент успешно создан для случая {case_id}")
                else:
                    print("\nОшибка при создании виртуального пациента.")
                    self.logger.error(f"Ошибка при создании виртуального пациента для случая {case_id}")
            except Exception as e:
                error_msg = f"Ошибка при создании виртуального пациента: {str(e)}"
                self.logger.error(error_msg)
                print(f"\nОшибка при создании виртуального пациента: {str(e)}")
            
        except Exception as e:
            error_msg = f"Ошибка при финализации данных: {str(e)}"
            self.logger.error(error_msg)
            print(f"\nОшибка при финализации данных: {str(e)}")

def generate_case_id():
    """
    Генерирует идентификатор случая со случайными цифрами и текущей датой
    
    Returns:
        str: Идентификатор случая в формате "case_RANDOM_DATE"
    """
    # Генерируем 6 случайных цифр
    random_digits = ''.join([str(random.randint(0, 9)) for _ in range(6)])
    
    # Получаем текущую дату в формате YYYYMMDD
    date_suffix = datetime.now().strftime('%Y%m%d')
    
    # Создаем идентификатор
    case_id = f"case_{random_digits}_{date_suffix}"
    
    return case_id

def multiline_input(prompt_text):
    """
    Многострочный ввод с помощью prompt_toolkit:
    - Enter — отправить
    - Esc+Enter или Ctrl+Enter — новая строка
    """
    bindings = KeyBindings()
    session = PromptSession()
    
    @bindings.add('enter')
    def _(event):
        buffer = event.app.current_buffer
        if buffer.complete_state:
            buffer.complete_next()
        else:
            event.app.exit(result=buffer.text)
    # Shift+Enter не поддерживается терминалом как отдельная комбинация
    # Для новой строки используйте Esc+Enter или Ctrl+Enter (стандарт prompt_toolkit)
    return session.prompt(prompt_text, multiline=True, key_bindings=bindings)

def main():
    # Создаем парсер аргументов
    parser = argparse.ArgumentParser(description='Система создания виртуальных пациентов')
    parser.add_argument('--debug', action='store_true', help='Включить потоковый вывод ответов языковых моделей')
    args = parser.parse_args()

    # === ГЛОБАЛЬНАЯ НАСТРОЙКА ЛОГИРОВАНИЯ ===
    setup_global_logging(args.debug)

    # Создаем агент-куратор с передачей режима отладки
    curator = CuratorAgent(debug_mode=args.debug)
    curator.start_conversation()

if __name__ == "__main__":
    main()