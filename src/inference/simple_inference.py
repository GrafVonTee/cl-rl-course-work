import torch
from unsloth import FastLanguageModel
from src.logger import setup_logger

logger = setup_logger(__name__, "inference.log")

def generate_response(model, tokenizer, prompt_text, max_new_tokens=250):
    FastLanguageModel.for_inference(model)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    logger.info(f"Запуск генерации (длина промпта: {len(prompt_text)} символов)")
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

    output_ids = generated_ids[0][len(inputs.input_ids[0]):]
    results = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    logger.info("Генерация завершена")
    return results
