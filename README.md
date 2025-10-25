# Model Fingerprinting

Test whether language models can be uniquely identified by their responses to out-of-distribution (OOD) queries.

## Concept

Different models (or the same model trained with different hyperparameters/data ordering) are unique functions. By querying them with out-of-distribution prompts (e.g., "8fs234ks2" or "purple cat people like"), we can capture their unique token prediction patterns as "fingerprints."

## Setup

```bash
# Install dependencies
uv sync

# Or if you haven't created the venv yet
uv venv
uv sync

# Generate OOD prompts (creates ood_prompts.json with 1000 prompts)
uv run python generate_ood_prompts.py
```

## Usage

### Basic Commands

**Test a single model (default: Llama-3.2-1B)**
```bash
uv run python code/fingerprint.py
```

**Test a specific model**
```bash
uv run python code/fingerprint.py meta-llama/Llama-3.2-3B
```

**Test all models**
```bash
uv run python code/fingerprint.py --all
```

**List available models**
```bash
uv run python code/fingerprint.py --list
```

### Advanced Options

**Custom temperature and seed**
```bash
# Default: greedy decoding (deterministic, T=0.0)
uv run python code/fingerprint.py --all

# Use sampling with temperature=1.0
uv run python code/fingerprint.py --all --temp 1.0

# Use higher temperature for more randomness
uv run python code/fingerprint.py --all --temp 1.5 --seed 123

# Generate more tokens
uv run python code/fingerprint.py --all --num-tokens 10
```

**Available arguments:**
- `--temperature, --temp, -t` - Sampling temperature (default: 0.0 for greedy, use 1.0 for sampling)
- `--seed, -s` - Random seed for reproducibility (default: 42)
- `--num-tokens, -n` - Number of tokens to generate (default: 5)
- `--prompts-file, -p` - Path to JSON file with prompts (default: ood_prompts.json)
- `--all` - Test all models
- `--list` - List available models

## Models Tested

1. **meta-llama/Llama-3.2-1B** - 1B parameter Llama model
2. **meta-llama/Llama-3.2-3B** - 3B parameter Llama model
3. **Qwen/Qwen2.5-1.5B** - 1.5B dense model
4. **Qwen/Qwen2.5-3B** - 3B dense model
5. **Qwen/Qwen1.5-MoE-A2.7B** - Mixture of Experts (~2.7B active)
6. **01-ai/Yi-1.5-6B** - 6B parameter Yi model
7. **allenai/OLMo-2-0425-1B** - 1B OLMo model
8. **EleutherAI/pythia-2.8b** - 2.8B Pythia model
9. **deepseek-ai/deepseek-coder-6.7b-base** - Code-specialized 6.7B model

## Output

Results are saved to `fingerprints_output/{timestamp}/` where timestamp is in format `YYYYMMDD_HHMMSS`:
- Individual JSON files for each model: `{model_name}.json`
- Combined results: `all_fingerprints.json`

Example directory structure:
```
fingerprints_output/
├── 20250125_143022/
│   ├── meta-llama_Llama-3.2-1B.json
│   ├── Qwen_Qwen2.5-1.5B.json
│   ├── ...
│   └── all_fingerprints.json
└── 20250125_150314/
    ├── ...
    └── all_fingerprints.json
```

Each fingerprint includes:
- The OOD prompt
- Generated text (5 tokens)
- Token IDs
- Model configuration (temperature, seed, etc.)

## OOD Prompts

### Generating Prompts

Run `generate_ood_prompts.py` to create 1000 OOD prompts:

```bash
uv run python generate_ood_prompts.py
```

This creates `ood_prompts.json` with 1000 prompts using various strategies:
- **Random strings**: `"8fs234ks2"`, `"xqz99ppml"`
- **Keyboard smashes**: `"asdfghjkl"`, `"qwertyuiop"`
- **Number/letter mixes**: `"123abc456def"`
- **Nonsensical phrases**: `"purple cat people like"`, `"rainbow elephants dance"`
- **Repeated patterns**: `"abcabc123123"`
- **CamelCase gibberish**: `"QuantumBanana"`, `"CrystalMemory"`
- **Special chars**: `"test_123-456.abc"`
- **All caps gibberish**: `"XYZQWERT"`
- **Hex-like strings**: `"a1b2c3d4e5f6"`
- **Base64-like strings**: `"dGVzdA=="`

### Custom Prompts File

You can provide your own prompts JSON file:

```bash
uv run python code/fingerprint.py --all --prompts-file my_prompts.json
```

**JSON format:**
```json
{
  "description": "My custom prompts",
  "count": 100,
  "prompts": [
    "prompt1",
    "prompt2",
    ...
  ]
}
```

## Configuration

**Default settings:**
- Temperature: 0.0 (greedy decoding, fully deterministic)
- Seed: 42
- Num tokens: 5

**Edit `code/fingerprint.py` to modify:**
- `OOD_PROMPTS` - List of prompts to test
- `TEST_MODELS` - List of models to fingerprint

**Note:** Temperature=0.0 gives fully deterministic greedy decoding. Use `--temp 1.0` with a fixed seed for reproducible but non-greedy sampling, which may better capture model differences on OOD prompts.
