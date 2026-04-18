"""
src/train/sft_lora.py
──────────────────────
QLoRA fine-tuning for Pocket-Agent.

Run from repo root (on Colab T4 GPU):
    python src/train/sft_lora.py

Input:  data/train_clean.jsonl
Output: artifacts/lora_adapter/
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "data/train_clean.jsonl"
OUTPUT_DIR = "./artifacts/lora_adapter"

# ── 4-bit QLoRA config ────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # T4: use float16, NOT bfloat16
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print(f"Loading base model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",   # avoids flash-attn dep issues on Colab
)
model.config.use_cache = False     # required for gradient checkpointing

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── LoRA config ───────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset = load_dataset("json", data_files=DATA_PATH, split="train")


def format_chat(example):
    """Apply Qwen2.5 chat template to each example."""
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


dataset = dataset.map(format_chat, remove_columns=["messages"])
print(f"Training on {len(dataset)} examples")
print("Sample:\n", dataset[0]["text"][:300])

# ── Training config (version-safe across all TRL releases) ───────────────────
import inspect

# Detect if SFTConfig accepts max_seq_length (TRL < 0.15) or not (TRL >= 0.15)
_sft_params = inspect.signature(SFTConfig.__init__).parameters
_has_max_seq = "max_seq_length" in _sft_params
_has_packing  = "packing" in _sft_params

_base_kwargs = dict(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,        # effective batch = 16
    learning_rate=3e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=True,                             # T4 supports fp16 natively, NOT bf16
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    dataset_text_field="text",
    report_to="none",                      # no wandb
)

if _has_max_seq:
    _base_kwargs["max_seq_length"] = 512
else:
    # Newer TRL: control length via tokenizer
    tokenizer.model_max_length = 512

if _has_packing:
    _base_kwargs["packing"] = False

sft_config = SFTConfig(**_base_kwargs)

# ── Trainer (version-safe) ────────────────────────────────────────────────────
_trainer_extra = {} if _has_max_seq else {"max_seq_length": 512}

for _tok_kwarg in ("processing_class", "tokenizer"):
    try:
        trainer = SFTTrainer(
            model=model,
            **{_tok_kwarg: tokenizer},
            args=sft_config,
            train_dataset=dataset,
            peft_config=lora_config,
            **_trainer_extra,
        )
        print(f"Trainer created with {_tok_kwarg}= ✅")
        break
    except TypeError:
        continue

print("Starting training…")
trainer.train()

print(f"Saving adapter → {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training complete ✅")

