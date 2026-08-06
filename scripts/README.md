# Judge Script

This tool uses **Alignment Ranker** models from the Qwen3.5 or **Gemma4** series to automatically evaluate how well two sets of responses (system‑generated vs. reference‑generated) align with a specific value, and outputs win/tie/lose statistics. An optional double‑check mechanism eliminates order bias.

------

## Features

- Load Hugging Face‑compatible Alignment Ranker models (**Qwen3.5 0.8B~9B and Gemma4 2B/4B**)
- Support JSON or JSONL input files
- Automatically identify common fields (`value`, `scenario`, `question`) and validate consistency
- Compare custom response fields (default: `response`)
- Optional `--double_check` mode: evaluate twice with swapped order; inconsistent results count as tie for higher robustness
- Built‑in definitions of 19 Schwartz values for prompt construction

------

## Dependencies

bash

```
pip install torch modelscope
```

> For GPU acceleration, ensure a CUDA‑compatible PyTorch version is installed. For Gemma4 models, `transformers` may be needed if loading directly from Hugging Face, but `modelscope` already handles it.

------

## Data Preparation

### System File & Reference File

Both files must be **JSON arrays** or **single JSON objects** (automatically converted to an array) and have the same number of records.
Each record must contain the following three common fields (field names are case‑insensitive and may have extra spaces):

- `value`: value name, must match one of the 19 built‑in keywords (see Appendix)
- `scenario`: scenario description
- `question`: question description

Additionally, each record must contain a response field (field name specified by `--system_field` and `--reference_field`, both default to `"response"`).

**Example (JSON array):**

```json
[
  {
    "value": "Achievement",
    "scenario": "Workplace competition",
    "question": "How to strive for a promotion?",
    "response": "I would set clear performance goals..."
  }
]
```



**Example (JSONL):** One JSON object per line.



------

## Usage

### Basic Command (without double‑check)

```bash
python judge.py --system system.json --reference reference.json
```



### Enable Double‑Check

```bash
python compare.py --system system.json --reference reference.json --double_check
```



### Specify Model and Field Names

bash

```bash
python compare.py \
    --model_name ctgee5/Gemma4-E4B-It-Alignment-Ranker \
    --system sys.jsonl \
    --reference ref.jsonl \
    --system_field output \
    --reference_field answer \
    --double_check
```

------

## Arguments

| Argument            | Type | Required | Default                         | Description                                       |
| :------------------ | :--- | :------- | :------------------------------ | :------------------------------------------------ |
| `--model_name`      | str  | No       | `Qwen3.5-0.8B-Alignment-Ranker` | Model name; see allowed values below              |
| `--system`          | str  | Yes      | –                               | Path to system file (`.json` or `.jsonl`)         |
| `--reference`       | str  | Yes      | –                               | Path to reference file (`.json` or `.jsonl`)      |
| `--system_field`    | str  | No       | `response`                      | Field name storing the response in system file    |
| `--reference_field` | str  | No       | `response`                      | Field name storing the response in reference file |
| `--double_check`    | flag | No       | `False`                         | Add this flag to enable double‑check              |

**Supported model names (12 total; can be used directly or with `ctgee5/` prefix):**

| Base Model                       | Prefixed Version                        |
| :------------------------------- | :-------------------------------------- |
| `Qwen3.5-0.8B-Alignment-Ranker`  | `ctgee5/Qwen3.5-0.8B-Alignment-Ranker`  |
| `Qwen3.5-2B-Alignment-Ranker`    | `ctgee5/Qwen3.5-2B-Alignment-Ranker`    |
| `Qwen3.5-4B-Alignment-Ranker`    | `ctgee5/Qwen3.5-4B-Alignment-Ranker`    |
| `Qwen3.5-9B-Alignment-Ranker`    | `ctgee5/Qwen3.5-9B-Alignment-Ranker`    |
| `Gemma4-E2B-It-Alignment-Ranker` | `ctgee5/Gemma4-E2B-It-Alignment-Ranker` |
| `Gemma4-E4B-It-Alignment-Ranker` | `ctgee5/Gemma4-E4B-It-Alignment-Ranker` |

------

## Output

After execution, two lines of statistics are printed:

```python
The number of unknown is 0.
The numbers of wins, ties, and losses are 12, 5, and 3 respectively.
```



- **win**: Number of times the system response aligns better with the value than the reference
- **tie**: Number of times both responses are equally aligned (or inconsistent in double‑check mode)
- **lose**: Number of times the system response aligns worse than the reference
- **unknown**: Number of times the model output could not be parsed as `win/tie/lose` (rare)

------

## Double‑Check Mode Explained

When `--double_check` is enabled, each record is evaluated twice:

1. Normal order: Response 1 = system, Response 2 = reference
2. Swapped order: Response 1 = reference, Response 2 = system (result then inverted)

If both evaluations agree (e.g., both yield `win`), that result is counted. If they differ, the result is counted as `tie`.
This reduces positional bias and improves evaluation reliability.

------

## Important Notes

- File encoding must be UTF‑8.
- The `value` field must exactly match one of the 19 built‑in keywords (see Appendix), otherwise a `KeyError` will occur. Please verify your data before running.
- If the input JSON is a single object (e.g., `{"value":"...", ...}`), it is automatically converted to a one‑element list.
- Model inference uses `device_map="auto"` by default, automatically assigning GPU/CPU.
- If `pad_token` is `None`, it is automatically set to `eos_token`.
- **Gemma4 models** use the same interface as Qwen; `trust_remote_code=True` is already set, no extra configuration needed.

------

## Appendix: Built‑in Value Keywords

python

```python
"Self-direction–thought"
"Self-direction–action"
"Stimulation"
"Hedonism"
"Achievement"
"Power–dominance"
"Power–resources"
"Face"
"Security–personal"
"Security–societal"
"Tradition"
"Conformity–rules"
"Conformity–interpersonal"
"Humility"
"Benevolence–dependability"
"Benevolence–caring"
"Universalism–concern"
"Universalism–nature"
"Universalism–tolerance"
```

Ensure the `value` field in your data exactly matches these strings (including hyphens and case).
