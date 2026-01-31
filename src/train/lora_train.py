import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel
import src.config as config
from src.logger import setup_logger


logger = setup_logger(__name__, "training.log")


def train_model(dataset_fn):
    logger.info(f"Загрузка модели {config.MODEL_PATH}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.MODEL_PATH,
        max_seq_length = config.MAX_TOKENS,
        dtype = None,
        load_in_4bit = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )

    logger.info("Составляем датасет...")
    dataset = dataset_fn(tokenizer, split="train")

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = config.MAX_TOKENS,
        packing = True,
        args = TrainingArguments(
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = 8,
            max_steps = 300,
            warmup_steps = 25,
            learning_rate = 2e-5,
            lr_scheduler_type="cosine",
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 25,
            output_dir = "checkpoints",
            optim = "adamw_8bit",
            report_to = "none", # Отключаем wandb
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
        ),
    )

    logger.info("Начинаем обучение...")
    trainer.train()

    logger.info(f"Сохраняем объединенную модель в {config.SFT_MODEL_PATH}...")
    model.save_pretrained(config.SFT_MODEL_PATH, tokenizer)
    tokenizer.save_pretrained(config.SFT_MODEL_PATH)

    logger.info("Обучение завершено!")

    return model, tokenizer
