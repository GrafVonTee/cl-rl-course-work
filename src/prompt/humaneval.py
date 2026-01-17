from src.config import DATASETS_DIR
from datasets import load_dataset


def get_dataset():
    return load_dataset("openai/openai_humaneval", cache_dir=DATASETS_DIR)


def get_prepared_dataset(tokenizer):
    humaneval = get_dataset()
    return humaneval.map(build_prompt, fn_kwargs={"tokenizer": tokenizer})


def build_prompt(example: dict, tokenizer) -> str:
    task_id = example.get("task_id", "")
    task_text = example["prompt"].rstrip()
    entry_point = example.get("entry_point", "")

    system_msg = (
        "You are an expert Python coding assistant. "
        "You will be given a Python file snippet containing imports and a single function "
        "signature with a docstring. Complete the function implementation so it is correct."
    )

    user_msg = (
        f"Task ID: {task_id}\n"
        f"Function to implement: {entry_point}\n\n"
        "Complete the following code by writing the function body.\n"
        "- Keep all existing imports, the function name, and its arguments unchanged.\n"
        "- Do not modify the docstring.\n"
        "- You may add local helper functions if needed, but do not change the target signature.\n"
        "- Return ONLY valid Python code (no explanations, no markdown, no code fences).\n\n"
        "Code:\n"
        f"{task_text}\n"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt
