"""
Менеджер API ключей и моделей для системы виртуальных пациентов.
Реализует паттерн "барабан револьвера" для API ключей и фолбэк для моделей.
"""

import time
import logging
import itertools
from openai import OpenAI, Stream
from config import GOOGLE_API_KEYS, MODELS, MAX_API_RETRIES, API_RETRY_TIMEOUT

class ApiManager:
    def __init__(self):
        """Инициализация менеджера API"""
        # Логирование
        self.logger = logging.getLogger("api_manager")
        # self.logger.setLevel(logging.INFO)  # Удалено, теперь уровень задается глобально

        
        # Проверка конфигурации
        if not GOOGLE_API_KEYS:
            raise ValueError("Не указаны API ключи в файле конфигурации")
            
        for model_type in ["dialogue", "synthesis", "arbitrator"]:
            if not MODELS.get(model_type):
                raise ValueError(f"Не указаны модели типа {model_type} в файле конфигурации")
    
    def execute_with_retry(self, model_type, api_call_func):
        """
        Выполняет API запрос с перебором всех доступных комбинаций моделей и API ключей
        
        Args:
            model_type: Тип модели ("dialogue", "synthesis" или "arbitrator")
            api_call_func: Функция, принимающая название модели и клиент API, и выполняющая API запрос
            
        Returns:
            Результат выполнения api_call_func или None в случае неудачи
        """
        # Получаем списки всех доступных моделей и API ключей
        all_models = MODELS.get(model_type, [])
        all_api_keys = GOOGLE_API_KEYS
        
        # Проверяем наличие моделей и ключей
        if not all_models or not all_api_keys:
            self.logger.error(f"Нет доступных моделей типа {model_type} или API ключей")
            return None
        
        # Создаем итератор для всех возможных комбинаций моделей и ключей
        combinations = list(itertools.product(all_models, all_api_keys))
        self.logger.info(f"Создан список из {len(combinations)} комбинаций модель/ключ для перебора")
        
        # Счетчик всех попыток
        total_attempts = 0
        # Множество для отслеживания уже использованных комбинаций
        tried_combinations = set()
        
        # Перебираем все возможные комбинации моделей и ключей
        for current_model, current_key in combinations:
            # Пропускаем уже использованные комбинации
            combo_key = f"{current_model}:{current_key[:8]}"
            if combo_key in tried_combinations:
                continue
                
            tried_combinations.add(combo_key)
            total_attempts += 1
            
            # Логируем попытку
            self.logger.info(f"Попытка {total_attempts}: модель {current_model} с ключом {current_key[:8]}...")
            
            try:
                # Создаем клиент с конкретным ключом
                client = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta", api_key=current_key)
                
                # Выполняем запрос с текущей моделью и клиентом
                response = api_call_func(model=current_model, client=client)
                
                # Проверяем, является ли ответ потоковым
                if isinstance(response, Stream) or 'Stream' in str(type(response)):
                    # Для потоковых ответов не проверяем наличие choices, просто возвращаем поток
                    self.logger.info(f"Успешный потоковый запрос к {current_model} с ключом {current_key[:8]}")
                    return response
                else:
                    # Для обычных ответов выполняем стандартные проверки
                    if response is None:
                        raise Exception("API вернул пустой ответ")
                    
                    if not hasattr(response, 'choices') or not response.choices:
                        raise Exception(f"Ответ API не содержит необходимых данных: {response}")
                    
                    self.logger.info(f"Успешный запрос к {current_model} с ключом {current_key[:8]}")
                    return response
                
            except Exception as e:
                self.logger.warning(f"Ошибка при запросе с моделью {current_model} и ключом API {current_key[:8]}...: {str(e)}")
                
                # Пауза перед следующей попыткой
                time.sleep(API_RETRY_TIMEOUT)
        
        # Если мы здесь, значит все комбинации перебраны и ни одна не сработала
        self.logger.error(f"Все {total_attempts} комбинаций моделей типа {model_type} и API ключей перебраны без успеха")
        return None

# Создаем глобальный экземпляр менеджера API
api_manager = ApiManager() 