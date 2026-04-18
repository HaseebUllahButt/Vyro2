"""
src/train/sft_lora.py
──────────────────────
QLoRA fine-tuning for Pocket-Agent.
Version-safe across TRL 0.12 – 0.18+.
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "data/train_clean.jsonl"
OUTPUT_DIR = "./artifacts/lora_adapter"

# ── 4-bit QLoRA config ────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print(f"Loading base model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)
model.config.use_cache = False

# Force ALL non-quantized params to float16.
# Qwen2.5 ships with bfloat16 layer norms — BnB doesn't quantize these,
# so they stay bf16 and crash the fp16 AMP gradient scaler on T4.
for name, param in model.named_parameters():
    if param.dtype == torch.bfloat16:
        param.data = param.data.to(torch.float16)
print("All bfloat16 params cast to float16 ✅")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.model_max_length = 512

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

# ── Detect TRL version and create trainer ─────────────────────────────────────
import trl
print(f"TRL version: {trl.__version__}")

from trl import SFTTrainer

# Use basic TrainingArguments — works everywhere, no API breakage
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
)

# Try each possible SFTTrainer signature until one works
ATTEMPTS = [
    # Attempt 1: Newest TRL (>=0.15) — processing_class, no max_seq_length
    dict(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    ),
    # Attempt 2: TRL 0.12–0.14 — processing_class + max_seq_length
    dict(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        max_seq_length=512,
        dataset_text_field="text",
        packing=False,
    ),
    # Attempt 3: Older TRL — tokenizer kwarg + max_seq_length
    dict(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        max_seq_length=512,
        dataset_text_field="text",
        packing=False,
    ),
    # Attempt 4: Minimal — just model and dataset
    dict(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    ),
]

trainer = None
for i, kwargs in enumerate(ATTEMPTS):
    try:
        trainer = SFTTrainer(**kwargs)
        print(f"Trainer created with attempt {i+1} ✅")
        break
    except TypeError as e:
        print(f"Attempt {i+1} failed: {e}")
        continue

if trainer is None:
    raise RuntimeError("All trainer creation attempts failed — check TRL version")

print("Starting training…")
trainer.train()

print(f"Saving adapter → {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training complete ✅")
