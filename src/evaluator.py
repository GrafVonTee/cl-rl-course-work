import json
import numpy as np
from tqdm.auto import tqdm
from typing import List, Dict
from vllm import SamplingParams

from src.executor import LocalExecutor
from src.metrics import BaseCodeMetric, ExecutionResult
from src.data.types import CodingTask
import src.config as config  # <--- 1. Импортируем конфиг

class Evaluator:
    def __init__(self, llm_engine, tokenizer, metrics: List[BaseCodeMetric]):
        self.llm = llm_engine
        self.tokenizer = tokenizer
        self.metrics = metrics
        self.executor = LocalExecutor()

    def run(self, tasks: List[CodingTask]) -> Dict[str, float]:
        final_results = {}

        # Группируем метрики и сразу готовим SamplingParams
        grouped_configs = self._group_metrics_and_prepare_params()

        for config_key, group in grouped_configs.items():
            sampling_params = group['params']
            metrics_in_group = group['metrics']

            # Логируем, с какими параметрами запускаем
            print(f"\n🚀 Group: {[m.name for m in metrics_in_group]}")
            print(f"⚙️ Params: n={sampling_params.n}, temp={sampling_params.temperature}, "
                  f"max_tokens={sampling_params.max_tokens}, logprobs={sampling_params.logprobs}")

            # --- PHASE A: GENERATION ---
            prompts = [t.prompt for t in tasks]

            # vLLM generate
            outputs = self.llm.generate(prompts, sampling_params)

            # --- PHASE B: EXECUTION ---
            # Нам нужно собрать плоский список всех сгенерированных сэмплов,
            # чтобы отдать их в ProcessPoolExecutor пачкой.

            # 1. Собираем задачи в список
            flat_tasks_input = []     # [(code, tests), ...]
            map_indices = []          # [(task_idx, sample_idx), ...] для восстановления структуры

            for i, request_output in enumerate(outputs):
                task_data = tasks[i]
                for j, sample in enumerate(request_output.outputs):
                    flat_tasks_input.append((sample.text, task_data.tests))
                    map_indices.append((i, j))

            # 2. ЗАПУСКАЕМ ПАРАЛЛЕЛЬНО (Вся магия тут)
            # Это вернет список ExecutionResult такой же длины, как flat_tasks_input
            flat_results = self.executor.batch_execute(flat_tasks_input)

            # 3. Восстанавливаем структуру (Task -> Samples)
            # Инициализируем пустой список списков
            all_exec_results = [[] for _ in range(len(tasks))]

            # Раскладываем результаты по полочкам и добавляем энтропию
            for k, exec_res in enumerate(flat_results):
                task_idx, sample_idx = map_indices[k]

                # Достаем энтропию, которую мы могли посчитать ранее или считаем сейчас
                # (В vLLM outputs хранятся в исходном объекте outputs)
                original_sample = outputs[task_idx].outputs[sample_idx]
                entropy = self._calculate_entropy(original_sample)
                exec_res.entropy = entropy

                all_exec_results[task_idx].append(exec_res)

            # --- PHASE C: METRICS ---
            for metric in metrics_in_group:
                score = metric.calculate(all_exec_results)
                final_results[metric.name] = score
                print(f"📊 {metric.name}: {score:.4f}")

        return final_results

    def _group_metrics_and_prepare_params(self):
        groups = {}

        # 2. Берем глобальные настройки из config.py
        base_settings = config.SAMPLING_SETTINGS.copy()

        # Удаляем из базы то, что контролируют метрики, чтобы не было конфликтов
        # (vLLM использует 'n' вместо 'num_return_sequences')
        base_settings.pop("n", None)
        base_settings.pop("temperature", None)

        for metric in self.metrics:
            # Ключ для группировки (уникальные настройки метрики)
            cfg = metric.gen_config
            cfg_key = json.dumps(cfg, sort_keys=True)

            if cfg_key not in groups:
                # 3. МЕРДЖИМ: Глобальный конфиг + Специфика метрики

                # Маппинг ключей: Metrics (HF style) -> vLLM style
                n = cfg.get("num_return_sequences", 1)
                temp = cfg.get("temperature", 0.0)

                # Создаем объект vLLM SamplingParams
                # Мы распаковываем **base_settings (там max_tokens, repetition_penalty и т.д.)
                # И явно задаем n и temperature
                vllm_params = SamplingParams(
                    n=n,
                    temperature=temp,
                    stop_token_ids=[self.tokenizer.eos_token_id], # Важно для остановки
                    **base_settings
                )

                groups[cfg_key] = {'params': vllm_params, 'metrics': []}

            groups[cfg_key]['metrics'].append(metric)

        return groups

    def _calculate_entropy(self, sample_output):
        """Считает энтропию для vLLM outputs"""
        # vLLM возвращает logprobs как список словарей {token_id: logprob}
        if not sample_output.logprobs:
            return 0.0

        entropies = []
        for step_logprobs in sample_output.logprobs:
            # step_logprobs: Dict[int, Logprob] (top-k tokens)
            if not step_logprobs: continue

            # Для точной энтропии нужны все вероятности, но vLLM дает топ-K.
            # Берем вероятность ВЫБРАННОГО токена как прокси уверенности.
            # Это упрощение, но работает для детекции галлюцинаций.

            # sample_output.token_ids содержит id выбранных токенов, но logprobs - это список по шагам.
            # vLLM устроена так: step_logprobs[token_id].logprob дает лог-вероятность.

            # Просто берем logprob того токена, который был выбран (он всегда есть в возврате, если logprobs=1)
            # Но проще взять values(), так как мы обычно просим logprobs=1, там будет 1 значение
            val = list(step_logprobs.values())[0].logprob

            # Entropy contribution ~ -log(p) (Surprise)
            entropies.append(-val)

        return np.mean(entropies) if entropies else 0.0
