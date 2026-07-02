"""Hyperparameters and identifiers shared between the Kaggle training
notebook and (eventually) the local inference wrapper. Kept here so a
re-run uses the same constants without copy-paste drift.
"""
from __future__ import annotations

# --- Model ---
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_REPO = "visethchapman/ercot-text2sql-qwen-1.5b-lora"  # HF Hub destination

# --- LoRA (QLoRA: 4-bit base + LoRA adapter) ---
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# --- Training ---
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4              # effective batch = 16
MAX_SEQ_LEN = 2048
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
LR_SCHEDULER = "cosine"
LOGGING_STEPS = 10
SAVE_STEPS = 100
EVAL_STEPS = 100

# --- Data ---
SYSTEM_PROMPT_KEY = "messages"  # SFTTrainer auto-detects chat-template format
