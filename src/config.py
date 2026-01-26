import os

MODELS_DIR = "./models/"
DATASETS_DIR = "./datasets/"
LOGS_DIR = "./logs/"
MODELS = {
	"0.6b": {
		"name": "Qwen/Qwen3-0.6B",
		"path": MODELS_DIR + "qwen3-0.6b"
	},
	"4b-instruct": {
		"name": "Qwen/Qwen3-4B-Instruct-2507",
		"path": MODELS_DIR + "qwen3-4B-instruct-2507"
	},
}
SELECTED_MODEL = "0.6b"
MODEL_NAME = MODELS[SELECTED_MODEL]["name"]
MODEL_PATH = MODELS[SELECTED_MODEL]["path"]
SFT_MODEL_PATH = f"{MODEL_PATH}-sft"
PTUNING_MODEL_PATH = f"{MODEL_PATH}-ptuning"

VLLM_PARAMS = {
	"max_model_len": 2048,
	"dtype": "auto",
	"gpu_memory_utilization": 0.7,
    "enforce_eager": False,
    "seed": 42,
    "enable_prefix_caching": False,
    "trust_remote_code": True,
}

SAMPLING_SETTINGS = {
    "max_tokens": 2048,
    "ignore_eos": False,
    "detokenize": True,
    "logprobs": 1,
    "repetition_penalty": 1,
}

NUM_PROCESSES = 8
