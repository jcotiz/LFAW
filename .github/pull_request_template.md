# 🚀 Pull Request - LFAW 

## 📌 What does this PR do?
Clearly describe what problem this PR solves or what optimization/experiment it introduces.

---

## 🏗 Affected Area

- [ ] 🧠 Model loading / architecture exploration (`explore_model.py`)
- [ ] ⚡ Quantization strategy (`measure_int4.py` / INT4 / INT8 / NF4)
- [ ] 📊 Inference speed benchmarking (`measure_velocity.py`)
- [ ] 🔧 Environment / configuration (`.env`, `HF_TOKEN`, dependencies)
- [ ] 🐍 Dependencies / `requirements.txt`
- [ ] 📝 Documentation / README

---

## 🔗 Related Issue
Closes #

---

## 🛠 Type of Change

- [ ] ✨ New quantization experiment
- [ ] 🐛 Bugfix
- [ ] ♻️ Refactor / code cleanup
- [ ] 📊 New benchmark / measurement
- [ ] 🔥 Performance improvement
- [ ] 🧱 New model or architecture support

---

## 🤖 Model & Quantization Details

| Field | Value |
|---|---|
| **Model** | e.g. `meta-llama/Llama-3.1-8B` |
| **Quantization** | e.g. INT4 / INT8 / NF4 / FP16 / None |
| **BitsAndBytes config** | e.g. `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"` |
| **Device** | e.g. `cuda` / `cpu` |
| **VRAM target** | ≤ 6 GB |

---

## 📊 Benchmark Results

Fill in before/after if applicable:

| Metric | Before | After |
|---|---|---|
| **Tokens generated** | | |
| **Total time (s)** | | |
| **Speed (tokens/sec)** | | |
| **VRAM usage (GB)** | | |
| **Weights loaded (it/s)** | | |

> Prompt used for testing: `"Artificial intelligence is"` — `NUM_TOKENS = 50`

---

## 🧪 How Has This Been Tested?

Explain the steps to reproduce or validate this change:

1. Set up `.env` with a valid `HF_TOKEN`
2. Activate `venv` and install dependencies
3. Run the relevant script: `py measure_int4.py` / `py measure_velocity.py`
4. Check console output for `=== RESULTS ===` block

---

## 📸 Evidence (Logs / Screenshots / Terminal Output)

Paste terminal output or attach screenshot. Example format:

```
=== INT4 RESULTS ===
Generated tokens:  50
Total time:        71.60 seconds
Speed:             0.70 tokens/sec
```

---

## ✅ Checklist

- [ ] Script runs without errors end-to-end
- [ ] Model loads correctly with the specified quantization config
- [ ] VRAM usage stays within the 6 GB target
- [ ] `HF_TOKEN` is read from `.env` and not hardcoded
- [ ] No sensitive files committed (`.env`, `venv/`, cache dirs)
- [ ] Benchmark results are reproducible with `do_sample=False` (greedy)
- [ ] `double_quant` and `compute_dtype` are explicitly set if using BnB

---

## 🧠 Technical Notes (Optional)

Relevant implementation details, architectural observations, or next steps.
e.g. layer shapes, parameter counts, attention head config, memory trade-offs.