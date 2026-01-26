import multiprocessing as mp
import signal
import sys
import re
from typing import List, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from src.metrics import ExecutionResult
import src.config as config

# --- УТИЛИТЫ ---

def extract_code_from_completion(text: str) -> str:
    """Вырезает код из блока ```python ... ``` или возвращает как есть."""
    text = re.sub(r"<think>[\s\S]*?(?:</think>|$)", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    if "def " in text:
        return text[text.find("def "):]
    return text

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout reached")

# --- ВОРКЕР (Функция верхнего уровня для Pickle) ---

def _process_single_sample(args: Tuple[str, List[str], float]) -> ExecutionResult:
    """
    Воркер, который выполняется в отдельном процессе пула.
    Принимает (generated_text, tests, timeout).
    """
    generated_text, tests, timeout = args

    # 1. Парсинг кода
    clean_code = extract_code_from_completion(generated_text)

    passed_count = 0
    total_count = len(tests)
    logs = ""

    # 2. Подготовка окружения
    # Мы используем signal.alarm для таймаута БЕЗ создания под-процессов.
    # Это работает только на Linux/Mac, но это в 100 раз быстрее mp.Process.
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout_handler)

    # 3. Исполнение тестов
    # Компилируем код один раз, чтобы поймать SyntaxError до запуска тестов
    try:
        compiled_code = compile(clean_code, "<string>", "exec")
    except Exception as e:
        return ExecutionResult(clean_code, 0, total_count, logs=f"Syntax Error: {e}")

    for test_case in tests:
        # Сбрасываем namespace для каждого теста, чтобы они не влияли друг на друга
        # Но оставляем импорты, если они были внутри compiled_code
        ns = {}

        try:
            # --- ЗАПУСК С ТАЙМАУТОМ ---
            if hasattr(signal, "SIGALRM"):
                signal.setitimer(signal.ITIMER_REAL, timeout)

            # 1. Исполняем код модели
            exec(compiled_code, ns, ns)
            # 2. Исполняем тест (assert ...)
            exec(test_case, ns, ns)

            # Если дошли сюда - успех
            if hasattr(signal, "SIGALRM"):
                signal.setitimer(signal.ITIMER_REAL, 0) # Отключаем таймер

            passed_count += 1

        except TimeoutException:
            # logs += f"Test timed out.\n"
            pass # Просто не засчитываем
        except Exception as e:
            # logs += f"Error: {e}\n"
            pass
        finally:
            # Гарантированно отключаем будильник
            if hasattr(signal, "SIGALRM"):
                signal.setitimer(signal.ITIMER_REAL, 0)

    return ExecutionResult(
        code=clean_code,
        passed_tests=passed_count,
        total_tests=total_count,
        logs=logs
    )


# --- КЛАСС ЭКЗЕКУТОРА ---

class LocalExecutor:
    def __init__(self, max_workers: int = None):
        # 2. ИСПОЛЬЗУЕМ КОНСТАНТУ ИЗ КОНФИГА
        # Если max_workers не передан явно, берем из config.NUM_PROCESSES.
        # Если и там нет (на всякий случай), фолбэк на cpu_count
        default_workers = getattr(config, "NUM_PROCESSES", mp.cpu_count())
        self.max_workers = max_workers if max_workers else default_workers

    def batch_execute(self,
                      tasks: List[Tuple[str, List[str]]],
                      timeout_per_test: float = 2.0) -> List[ExecutionResult]:
        """
        Параллельный запуск.
        """
        # Готовим аргументы
        map_args = [(text, tests, timeout_per_test) for text, tests in tasks]

        print(f"🚀 Executing {len(tasks)} samples in parallel using {self.max_workers} workers...")

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # list(executor.map(...)) гарантирует порядок результатов
            results = list(tqdm(
                executor.map(_process_single_sample, map_args),
                total=len(map_args),
                desc="Running Tests"
            ))

        return results

    def execute(self, generated_text: str, tests: List[str]) -> ExecutionResult:
        """Совместимость для одиночного запуска"""
        return _process_single_sample((generated_text, tests, 2.0))
