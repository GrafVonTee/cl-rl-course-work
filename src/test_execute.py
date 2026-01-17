import multiprocessing as mp
import queue as queue_mod
import re
from typing import List, Optional


def extract_code_from_completion(text: str) -> str:
    """Вырезает код из блока ```python ... ``` или возвращает как есть."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


def extract_tests(test_str: str) -> List[str]:
    """Парсит asserts из строки тестов (для HumanEval)."""
    if not test_str:
        return []

    lines = test_str.splitlines()
    tests: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*assert\b", line):
            buf = [line.rstrip()]
            i += 1
            # Обработка многострочных assert
            text = "\n".join(buf)
            balance = text.count("(") - text.count(")")
            balance += text.count("[") - text.count("]")
            balance += text.count("{") - text.count("}")

            while i < len(lines) and balance > 0:
                buf.append(lines[i].rstrip())
                text = "\n".join(buf)
                balance = text.count("(") - text.count(")")
                balance += text.count("[") - text.count("]")
                balance += text.count("{") - text.count("}")
                i += 1
            tests.append("\n".join(buf).strip())
        else:
            i += 1
    return tests


def _run_code_and_test_worker(result_queue: mp.Queue, code: str, test: str):
    """Воркер для безопасного исполнения кода."""
    ns = {}
    try:
        # ВАЖНО: exec небезопасен для продакшена, но ок для локальных тестов
        exec(code, ns, ns)
        exec(test, ns, ns)
        result_queue.put(True)
    except Exception:
        result_queue.put(False)


def run_single_test_with_timeout(code: str, test: str, timeout_sec: float = 1.0) -> bool:
    """Запускает тест в отдельном процессе с таймаутом."""
    result_queue: mp.Queue = mp.Queue()
    p = mp.Process(target=_run_code_and_test_worker, args=(result_queue, code, test))
    p.start()

    try:
        passed = result_queue.get(timeout=timeout_sec)
    except queue_mod.Empty:
        passed = False
    finally:
        if p.is_alive():
            p.terminate()
        p.join()

    return bool(passed)


def run_mbpp_tests_for_sample(raw_text: str, test_list: list) -> tuple[int, int]:
    code = extract_code_from_completion(raw_text)
    num_passed = 0
    for t in test_list:
        if run_single_test_with_timeout(code, t):
            num_passed += 1
    return num_passed, len(test_list)


def run_humaneval_tests_for_sample(raw_text: str, test_str: str, entry_point: str) -> tuple[int, int]:
    code = extract_code_from_completion(raw_text)
    tests = extract_tests(test_str)
    num_passed = 0
    full_code_context = f"candidate = {entry_point}\n"

    for t in tests:
        if run_single_test_with_timeout(code, full_code_context + t):
            num_passed += 1
    return num_passed, len(tests)
