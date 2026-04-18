#!/bin/bash
# src/quantize/quantize.sh
# ─────────────────────────────────────────────────────────────────────────────
# Merges LoRA adapter, converts to GGUF, and quantizes.
# Run from repo root on Colab after training:
#   bash src/quantize/quantize.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "════════════════════════════════════════"
echo " Step 1: Merge LoRA adapter into base weights"
echo "════════════════════════════════════════"

python - <<'PYEOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading base model (CPU fp16)…")
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)

print("Loading LoRA adapter…")
model = PeftModel.from_pretrained(base, "./artifacts/lora_adapter")

print("Merging and unloading…")
model = model.merge_and_unload()
model.save_pretrained("./artifacts/merged_model")

tok = AutoTokenizer.from_pretrained("./artifacts/lora_adapter", trust_remote_code=True)
tok.save_pretrained("./artifacts/merged_model")
print("Merge complete ✅")
PYEOF

echo ""
echo "════════════════════════════════════════"
echo " Step 2: Convert to GGUF (fp16)"
echo "════════════════════════════════════════"

python llama.cpp/convert_hf_to_gguf.py ./artifacts/merged_model \
    --outfile ./artifacts/model_f16.gguf \
    --outtype f16

echo "GGUF fp16 created ✅"
echo ""

echo "════════════════════════════════════════"
echo " Step 3: Quantize"
echo "════════════════════════════════════════"

# Primary: Q3_K_M (~210MB) — targets ≤250MB bonus gate
./llama.cpp/llama-quantize \
    ./artifacts/model_f16.gguf \
    ./artifacts/model_q3km.gguf \
    Q3_K_M

# Fallback: Q4_K_M (~280MB) — if Q3 quality is insufficient
./llama.cpp/llama-quantize \
    ./artifacts/model_f16.gguf \
    ./artifacts/model_q4km.gguf \
    Q4_K_M

echo ""
echo "File sizes:"
ls -lh ./artifacts/model_q3km.gguf ./artifacts/model_q4km.gguf

# Automatically pick Q3_K_M if under 250MB, else Q4_K_M
# (evaluation will confirm quality — see src/eval/score.py)
Q3_SIZE=$(stat -c%s ./artifacts/model_q3km.gguf 2>/dev/null || stat -f%z ./artifacts/model_q3km.gguf)

if [ "$Q3_SIZE" -lt 250000000 ]; then
    cp ./artifacts/model_q3km.gguf ./artifacts/model.gguf
    echo "Selected Q3_K_M as artifacts/model.gguf ($(du -h artifacts/model.gguf | cut -f1))"
else
    cp ./artifacts/model_q4km.gguf ./artifacts/model.gguf
    echo "Q3_K_M exceeds 250MB — using Q4_K_M as artifacts/model.gguf ($(du -h artifacts/model.gguf | cut -f1))"
fi

echo ""
echo "Quantization complete ✅"
echo "Run: python src/eval/score.py to verify accuracy"
