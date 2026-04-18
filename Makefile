.PHONY: all data train eval quantize demo test clean

all: data train eval quantize

data:
	python src/data/generate.py
	python src/data/lint.py

train:
	python src/train/sft_lora.py

eval:
	python src/eval/score.py

eval-verbose:
	python src/eval/score.py --verbose

quantize:
	bash src/quantize/quantize.sh

demo:
	python demo/app.py

demo-share:
	python demo/app.py --share

test:
	python test_single.py --interactive

gates:
	@echo "Running all hard gate checks..."
	@echo ""
	@echo "1. Network import scan:"
	@grep -n "import requests\|import urllib\|import http\|import socket\|import httpx" inference.py \
		&& echo "   FAIL" || echo "   PASS"
	@echo ""
	@echo "2. Model file size:"
	@ls -lh artifacts/model.gguf 2>/dev/null || echo "   MISSING — run make quantize first"
	@echo ""
	@echo "3. Latency benchmark (CPU):"
	@python - <<'EOF'
import time, sys
sys.path.insert(0, ".")
from inference import run
prompts = [
    "weather in Tokyo in Celsius?",
    "Convert 100 USD to EUR",
    "convert 5 km to miles",
    "What do I have tomorrow?",
    "tell me a joke",
] * 4
times = []
for p in prompts:
    t = time.time()
    run(p, [])
    times.append((time.time() - t) * 1000)
mean = sum(times)/len(times)
print(f"   Mean: {mean:.1f}ms  Max: {max(times):.1f}ms")
print("   PASS" if mean < 200 else "   FAIL — exceeds 200ms")
EOF

clean:
	rm -rf data/ artifacts/merged_model artifacts/model_f16.gguf \
	       artifacts/model_q3km.gguf artifacts/model_q4km.gguf
	@echo "Cleaned intermediate artifacts (adapter and model.gguf preserved)"
