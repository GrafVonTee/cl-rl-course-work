from functools import partial
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import src.config as config

def setup_model(use_sft=False):
    model_path = config.MODEL_PATH
    adapter_path = config.SFT_MODEL_PATH

    llm = LLM(
        model=model_path,
        enable_lora=use_sft,
        max_lora_rank=64 if use_sft else None,
        **config.VLLM_PARAMS
    )

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path if use_sft else model_path,
        trust_remote_code=True
    )

    if use_sft:
        request = LoRARequest("sft", 1, lora_path=adapter_path)
        llm.generate = partial(llm.generate, lora_request=request)

    sampling_params = SamplingParams(
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.pad_token_id],
        **config.SAMPLING_SETTINGS
    )

    return llm, tokenizer, sampling_params
