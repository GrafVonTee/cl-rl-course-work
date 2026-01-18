import time
from tqdm.auto import tqdm
import src.prompt.mbpp as mbpp
import src.prompt.humaneval as humaneval
import src.test_execute as test_execute
import src.metrics as metrics_utils


def run_mbpp_bench(mbpp_dataset, split, llm, tokenizer, sampling_params, batch_size=2048):
    dataset = mbpp_dataset[split]
    tests = dataset["test_list"]

    outputs_all = []
    metrics_all = []

    print(f"Starting MBPP benchmark on {len(dataset)} samples...")

    pbar = tqdm(range(0, len(dataset), batch_size))
    for i in pbar:
        batch_slice = dataset.select(range(i, min(i + batch_size, len(dataset))))

        # 1. Генерация
        prompts = [mbpp.build_prompt(x, tokenizer, train=False)["text"] for x in batch_slice]

        start_time = time.time()
        outs = llm.generate(prompts, sampling_params)
        duration = time.time() - start_time

        total_tokens = sum(len(out.outputs[0].token_ids) for out in outs)
        speed = total_tokens / duration if duration > 0 else 0
        avg_len = total_tokens / len(outs) if outs else 0

        pbar.set_postfix({"tok/s": f"{speed:.2f}", "avg_len": f"{avg_len:.1f}"})

        # 2. Тестирование
        for j, out in enumerate(outs):
            global_idx = i + j
            generated_text = out.outputs[0].text
            current_tests = tests[global_idx]

            passed, total = test_execute.run_mbpp_tests_for_sample(generated_text, current_tests)

            met = {
                "pass@1": 1 if passed == total else 0,
                "%passed": passed / total if total > 0 else 0,
                "entropy": metrics_utils.calculate_entropy_from_logprobs(out.outputs[0])
            }
            metrics_all.append(met)

        outputs_all.extend(outs)

    return outputs_all, metrics_all


def run_humaneval_bench(he_dataset, split, llm, tokenizer, sampling_params, batch_size=32):
    dataset = he_dataset[split]

    outputs_all = []
    metrics_all = []

    print(f"Starting HumanEval benchmark on {len(dataset)} samples...")

    pbar = tqdm(range(0, len(dataset), batch_size))
    for i in pbar:
        batch_slice = dataset.select(range(i, min(i + batch_size, len(dataset))))

        # 1. Генерация
        prompts = [humaneval.build_prompt(x, tokenizer) for x in batch_slice]

        start_time = time.time()
        outs = llm.generate(prompts, sampling_params)
        duration = time.time() - start_time

        total_tokens = sum(len(out.outputs[0].token_ids) for out in outs)
        speed = total_tokens / duration if duration > 0 else 0
        avg_len = total_tokens / len(outs) if outs else 0

        pbar.set_postfix({"tok/s": f"{speed:.2f}", "avg_len": f"{avg_len:.1f}"})

        # 2. Тестирование
        for j, out in enumerate(outs):
            generated_text = out.outputs[0].text
            test_str = batch_slice[j]["test"]
            entry_point = batch_slice[j]["entry_point"]

            passed, total = test_execute.run_humaneval_tests_for_sample(generated_text, test_str, entry_point)

            met = {
                "pass@1": 1 if passed == total else 0,
                "%passed": passed / total if total > 0 else 0,
                "entropy": metrics_utils.calculate_entropy_from_logprobs(out.outputs[0])
            }
            metrics_all.append(met)

        outputs_all.extend(outs)

    return outputs_all, metrics_all
