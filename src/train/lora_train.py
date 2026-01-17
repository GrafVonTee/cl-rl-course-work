import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel
import src.config as config
import src.prompt.mbpp as mbpp
from src.logger import setup_logger


logger = setup_logger(__name__, "training.log")

def train_model():
    """
    Загружает модель, обучает её и сохраняет результат.
    Возвращает (model, tokenizer) для немедленного тестирования.
    """
    logger.info(f"Загрузка модели {config.MODEL_PATH}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.MODEL_PATH,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    # Навешиваем адаптеры LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 64,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 64,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )

    logger.info("Подготовка датасета...")
    dataset = mbpp.get_prepared_dataset(tokenizer, split="train")

    # Конфигурация тренера
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 4,
            max_steps = 100,
            learning_rate = 3e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            output_dir = "checkpoints",
            optim = "adamw_8bit",
            report_to = "none", # Отключаем wandb
        ),
    )

    logger.info("Начинаем обучение...")
    trainer.train()

    logger.info(f"Сохраняем объединенную модель в {config.SFT_MODEL_PATH}...")
    model.save_pretrained(config.SFT_MODEL_PATH, tokenizer)
    tokenizer.save_pretrained(config.SFT_MODEL_PATH)

    logger.info("Обучение завершено!")

    return model, tokenizer
