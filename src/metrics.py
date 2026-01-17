import torch

def calculate_entropy_from_logprobs(seq) -> float:
    logprobs_list = []
    if not seq.logprobs:
        return 0.0

    for token_logprobs in seq.logprobs:
        # vLLM возвращает dict {token_id: LogprobObj}
        # Берем логпроб выбранного токена
        for token_id, logprob_obj in token_logprobs.items():
            if hasattr(logprob_obj, 'logprob'):
                val = logprob_obj.logprob
            else:
                val = float(logprob_obj)
            logprobs_list.append(val)
            break # Берем только топ-1 (реально выбранный)

    if not logprobs_list:
        return 0.0

    logprobs_tensor = torch.tensor(logprobs_list, dtype=torch.float)
    entropy = -logprobs_tensor.mean().item()
    return entropy

def aggregate_metrics(metrics_list: list):
    """Считает средние показатели."""
    if not metrics_list:
        return {"pass@1": 0, "%passed": 0}

    mean_pass1 = sum(x["pass@1"] for x in metrics_list) / len(metrics_list)
    mean_pct = sum(x["%passed"] for x in metrics_list) / len(metrics_list)

    return {
        "mean_pass@1": mean_pass1,
        "mean_%_passed": mean_pct,
        "total_samples": len(metrics_list)
    }
