"""Model Fingerprinting using vLLM.

This script queries language models with out-of-distribution prompts to generate
unique fingerprints based on their token predictions.
"""

from vllm import LLM, SamplingParams
from typing import List, Dict, Optional
import json
import os


class ModelFingerprinter:
    """Query models with OOD inputs to generate unique fingerprints."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B",
                 tensor_parallel_size: int = 1,
                 dtype: str = "auto"):
        """Initialize the model for fingerprinting.

        Args:
            model_name: HuggingFace model name or path
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            dtype: Data type for model weights ("auto", "float16", "bfloat16", etc.)
        """
        print(f"\nLoading model: {model_name}...")
        print(f"  - Tensor parallel size: {tensor_parallel_size}")
        print(f"  - Data type: {dtype}")
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=True  # Required for some models
        )
        print("✓ Model loaded successfully!\n")

    def fingerprint(self,
                   prompts: List[str],
                   num_tokens: int = 5,
                   temperature: float = 1.0,
                   top_p: float = 1.0,
                   seed: Optional[int] = None) -> List[Dict]:
        """Generate fingerprints for given prompts.

        Args:
            prompts: List of out-of-distribution prompts to query
            num_tokens: Number of tokens to generate per prompt (default: 5)
            temperature: Sampling temperature (1.0 = normal, 0.0 = greedy)
            top_p: Nucleus sampling parameter
            seed: Random seed for reproducibility

        Returns:
            List of dictionaries containing prompt, generated tokens, and metadata
        """
        print(f"  Generating {num_tokens} tokens for {len(prompts)} prompts...")
        print(f"  - Temperature: {temperature}")
        print(f"  - Top-p: {top_p}")
        print(f"  - Seed: {seed}")

        # Configure sampling parameters for exact token count
        sampling_params = SamplingParams(
            n=1,  # Number of output sequences
            max_tokens=num_tokens,
            min_tokens=num_tokens,  # Ensure we get exactly num_tokens
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )

        # Generate completions
        print(f"  Running inference...")
        outputs = self.llm.generate(prompts, sampling_params)
        print(f"  ✓ Inference complete!\n")

        # Format results
        results = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            token_ids = output.outputs[0].token_ids

            results.append({
                "prompt": prompt,
                "generated_text": generated_text,
                "token_ids": token_ids,
                "num_tokens": len(token_ids),
                "model": self.model_name,
            })

        return results

    def print_results(self, results: List[Dict]):
        """Pretty print fingerprinting results."""
        print(f"\n{'='*80}")
        print(f"Model: {self.model_name}")
        print(f"{'='*80}\n")

        for i, result in enumerate(results, 1):
            print(f"Query {i}:")
            print(f"  Prompt: '{result['prompt']}'")
            print(f"  Generated: '{result['generated_text']}'")
            print(f"  Token IDs: {result['token_ids']}")
            print(f"  Num tokens: {result['num_tokens']}")
            print()


# Fallback OOD prompts (used if file not found)
DEFAULT_OOD_PROMPTS = [
    "8fs234ks2",
    "purple cat people like",
    "xqz99ppml",
    "rainbow elephants dance",
    "kkk999jjj444",
    "quantum banana philosophy",
    "7h3_qu1ck_br0wn",
    "crystalline memory patterns",
    "zxcvbnmasdfghjkl",
    "ethereal computational dreams",
]


def load_ood_prompts(prompts_file: str = "ood_prompts.json") -> List[str]:
    """Load OOD prompts from JSON file.

    Args:
        prompts_file: Path to JSON file containing prompts

    Returns:
        List of prompt strings
    """
    if not os.path.exists(prompts_file):
        print(f"Warning: Prompts file '{prompts_file}' not found. Using default prompts.")
        return DEFAULT_OOD_PROMPTS

    try:
        with open(prompts_file, 'r') as f:
            data = json.load(f)
            prompts = data.get('prompts', [])
            print(f"Loaded {len(prompts)} prompts from {prompts_file}")
            return prompts
    except Exception as e:
        print(f"Error loading prompts file: {e}")
        print("Using default prompts instead.")
        return DEFAULT_OOD_PROMPTS


def load_models(models_file: str = "models.json") -> List[str]:
    """Load model names from JSON file.

    Args:
        models_file: Path to JSON file containing model configurations

    Returns:
        List of model name strings
    """
    # Default models (fallback)
    default_models = [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B",
        "Qwen/Qwen1.5-MoE-A2.7B",
        "01-ai/Yi-1.5-6B",
        "allenai/OLMo-2-0425-1B",
        "EleutherAI/pythia-2.8b",
        "deepseek-ai/deepseek-coder-6.7b-base",
    ]

    if not os.path.exists(models_file):
        print(f"Warning: Models file '{models_file}' not found. Using default models.")
        return default_models

    try:
        with open(models_file, 'r') as f:
            data = json.load(f)
            models_list = data.get('models', [])
            model_names = [m['name'] if isinstance(m, dict) else m for m in models_list]
            print(f"Loaded {len(model_names)} models from {models_file}")
            return model_names
    except Exception as e:
        print(f"Error loading models file: {e}")
        print("Using default models instead.")
        return default_models


# Load models from JSON file (check parent directory if not found)
def _get_models_file_path():
    """Find models.json in current or parent directory."""
    if os.path.exists("models.json"):
        return "models.json"
    elif os.path.exists("../models.json"):
        return "../models.json"
    else:
        return "models.json"  # Will use defaults

TEST_MODELS = load_models(_get_models_file_path())


def fingerprint_model(model_name: str,
                      prompts: List[str],
                      num_tokens: int = 5,
                      temperature: float = 0.0,
                      seed: int = 42) -> Dict:
    """Fingerprint a single model with given prompts.

    Args:
        model_name: HuggingFace model name
        prompts: List of OOD prompts
        num_tokens: Number of tokens to generate
        temperature: Sampling temperature (0.0 = greedy/deterministic, 1.0 = sampling)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with model name and fingerprint results
    """

    print("\n" + "="*80)
    print(f"FINGERPRINTING: {model_name}")
    print("="*80)

    # Initialize model
    fingerprinter = ModelFingerprinter(
        model_name=model_name,
        dtype="auto"
    )

    # Generate fingerprints
    print(f"\n[2/3] Generating fingerprints...")
    results = fingerprinter.fingerprint(
        prompts=prompts,
        num_tokens=num_tokens,
        temperature=temperature,
        seed=seed
    )

    # Print results
    print(f"[3/3] Results:")
    fingerprinter.print_results(results)

    return {
        "model": model_name,
        "fingerprints": results,
        "config": {
            "num_tokens": num_tokens,
            "temperature": temperature,
            "seed": seed,
        }
    }


def main():
    """Run model fingerprinting experiments."""
    import argparse
    import os
    from datetime import datetime

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Model Fingerprinting: Test models with OOD prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fingerprint.py                              # Test single model (default, T=0.0)
  python fingerprint.py --all                        # Test all models with greedy decoding
  python fingerprint.py --list                       # List available models
  python fingerprint.py meta-llama/Llama-3.2-3B     # Test specific model
  python fingerprint.py --all --temp 1.0             # Use sampling instead of greedy
  python fingerprint.py --all --seed 123 --temp 1.5  # Custom seed and temperature
        """
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Model name to fingerprint (default: meta-llama/Llama-3.2-1B)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all models in TEST_MODELS list"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models and exit"
    )
    parser.add_argument(
        "--temperature", "--temp", "-t",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for greedy, use 1.0 for sampling)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--num-tokens", "-n",
        type=int,
        default=5,
        help="Number of tokens to generate (default: 5)"
    )
    parser.add_argument(
        "--prompts-file", "-p",
        type=str,
        default="ood_prompts.json",
        help="Path to JSON file containing OOD prompts (default: ood_prompts.json)"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("MODEL FINGERPRINTING EXPERIMENT")
    print("="*80 + "\n")

    # Handle --list flag
    if args.list:
        print("Available models:")
        for i, model in enumerate(TEST_MODELS, 1):
            print(f"  {i}. {model}")
        return

    # Determine which models to test
    if args.all:
        models_to_test = TEST_MODELS
        print(f"Testing all {len(TEST_MODELS)} models")
    elif args.model:
        models_to_test = [args.model]
        print(f"Testing single model: {args.model}")
    else:
        # Default: just test Llama-3.2-1B
        models_to_test = ["meta-llama/Llama-3.2-1B"]
        print(f"Testing single model (default): {models_to_test[0]}")
        print(f"Use --all to test all models, or specify a model name")
        print(f"Use --list to see available models")

    # Load OOD prompts
    ood_prompts = load_ood_prompts(args.prompts_file)

    print(f"\nConfiguration:")
    print(f"  Temperature: {args.temperature}")
    print(f"  Seed: {args.seed}")
    print(f"  Num tokens: {args.num_tokens}")
    print(f"  Num prompts: {len(ood_prompts)}")
    print(f"  Prompts file: {args.prompts_file}\n")

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("fingerprints_output", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Run fingerprinting on each model
    all_results = []
    for i, model_name in enumerate(models_to_test, 1):
        print(f"\n{'='*80}")
        print(f"MODEL {i}/{len(models_to_test)}")
        print(f"{'='*80}")

        try:
            result = fingerprint_model(
                model_name=model_name,
                prompts=ood_prompts,
                num_tokens=args.num_tokens,
                temperature=args.temperature,
                seed=args.seed
            )
            all_results.append(result)

            # Save individual model results
            safe_name = model_name.replace("/", "_")
            output_file = os.path.join(output_dir, f"{safe_name}.json")
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n✓ Results saved to: {output_file}")

        except Exception as e:
            print(f"\n✗ Error fingerprinting {model_name}: {e}")
            continue

    # Save combined results
    combined_file = os.path.join(output_dir, "all_fingerprints.json")
    with open(combined_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "num_models": len(all_results),
            "models": all_results
        }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully fingerprinted {len(all_results)}/{len(models_to_test)} models")
    print(f"Combined results saved to: {combined_file}\n")


if __name__ == "__main__":
    main()
