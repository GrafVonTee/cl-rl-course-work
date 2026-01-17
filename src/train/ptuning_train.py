import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
import src.config as config
import src.prompt.mbpp as mbpp
from src.logger import setup_logger

logger = setup_logger(__name__, "ptuning.log")

# Наш кастомный тренер (без изменений)
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)

        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        else:
            logits = outputs.get("logits")
            labels = inputs.get("labels")

            # Проверка на случай, если labels всё еще нет
            if labels is None:
                raise ValueError("Labels are missing in inputs! Tokenization failed to create them.")

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return (loss, outputs) if return_outputs else loss

def train():
    logger.info(f"Load Base Model: {config.MODEL_NAME}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Setup P-Tuning...")
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text="You are an expert Python assistant.",
        tokenizer_name_or_path=config.MODEL_PATH,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    logger.info("Tokenizing...")
    raw_dataset = mbpp.get_prepared_dataset(tokenizer, split="train")

    # === ИСПРАВЛЕННАЯ ФУНКЦИЯ ===
    def tokenize_lazy(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            padding=False
        )
        # !!! ВАЖНО: Копируем input_ids в labels !!!
        # Без этого модель не знает, с чем сравнивать предсказания
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    # ============================

    tokenized_dataset = raw_dataset.map(
        tokenize_lazy,
        batched=True,
        remove_columns=raw_dataset.column_names
    )

    logger.info("Start Training...")

    trainer = CustomTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=TrainingArguments(
            output_dir=config.PTUNING_MODEL_PATH,
            max_steps=100,
            learning_rate=1e-2,
            logging_steps=10,
            report_to="none",
            fp16=True,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
        ),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, model=model)
    )

    trainer.train()

    model.save_pretrained(config.PTUNING_MODEL_PATH)
    logger.info(f"Saved to {config.PTUNING_MODEL_PATH}")

    return model, tokenizer

if __name__ == "__main__":
    train()
