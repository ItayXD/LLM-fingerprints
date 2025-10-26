#!/usr/bin/env python3
"""
Batch generate training data for all models in models.json.
Runs generate_training_data.py for each model, saving to distillation_text/{model_name}/
"""

import argparse
import subprocess
import json
import os
from datetime import datetime
from typing import List, Dict


def load_models(models_file: str = "models.json") -> List[Dict]:
    """Load models from JSON file."""
    # Check if models file exists in current dir or parent dir
    if not os.path.exists(models_file):
        if os.path.exists(f"../{models_file}"):
            models_file = f"../{models_file}"
        else:
            raise FileNotFoundError(f"Cannot find {models_file}")

    with open(models_file, 'r') as f:
        data = json.load(f)
        return data.get('models', [])


def run_generation(
    model_name: str,
    num_sequences: int,
    max_length: int,
    temperature: float,
    top_p: float,
    num_unique_prompts: int,
    prompt_seed: int,
    prompts_file: str,
    output_dir: str
) -> bool:
    """Run generate_training_data.py for a single model.

    Returns:
        True if successful, False otherwise
    """
    # Create model-specific output directory
    model_safe_name = model_name.replace("/", "_")
    model_output_dir = os.path.join(output_dir, model_safe_name)

    # Determine the correct path to generate_training_data.py
    # Check if we're in the code/ directory or parent directory
    if os.path.exists("generate_training_data.py"):
        script_path = "generate_training_data.py"
    elif os.path.exists("code/generate_training_data.py"):
        script_path = "code/generate_training_data.py"
    else:
        print(f"Error: Cannot find generate_training_data.py")
        return False

    # Build command
    cmd = [
        "python", script_path,
        "--model", model_name,
        "--num-sequences", str(num_sequences),
        "--max-length", str(max_length),
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--output-dir", model_output_dir,
    ]

    if num_unique_prompts is not None:
        cmd.extend(["--num-unique-prompts", str(num_unique_prompts)])

    if prompt_seed is not None:
        cmd.extend(["--prompt-seed", str(prompt_seed)])

    if prompts_file:
        # Resolve prompts file path (check current dir or parent dir)
        if os.path.exists(prompts_file):
            resolved_prompts = prompts_file
        elif os.path.exists(f"../{prompts_file}"):
            resolved_prompts = f"../{prompts_file}"
        else:
            print(f"Warning: Prompts file {prompts_file} not found, skipping")
            resolved_prompts = None

        if resolved_prompts:
            cmd.extend(["--prompts-file", resolved_prompts])

    print(f"\n{'='*80}")
    print(f"GENERATING DATA FOR: {model_name}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        # Run the command
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error generating data for {model_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate training data for all models in models.json"
    )
    parser.add_argument(
        "--models-file",
        type=str,
        default="models.json",
        help="Path to models JSON file (default: models.json)"
    )
    parser.add_argument(
        "--num-sequences", "-n",
        type=int,
        default=1000,
        help="Number of sequences per model (default: 1000)"
    )
    parser.add_argument(
        "--max-length", "-l",
        type=int,
        default=512,
        help="Maximum length per sequence (default: 512)"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    parser.add_argument(
        "--top-p", "-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p (default: 0.95)"
    )
    parser.add_argument(
        "--num-unique-prompts",
        type=int,
        default=None,
        help="Number of unique prompts to generate (default: min(1000, num_sequences))"
    )
    parser.add_argument(
        "--prompt-seed",
        type=int,
        default=42,
        help="Random seed for prompt generation (default: 42, use None for random)"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="JSON file with prompts to seed generation (optional, overrides diverse prompts)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="distillation_text",
        help="Base output directory (default: distillation_text)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to generate for (default: all from models.json)"
    )

    args = parser.parse_args()

    # Load models
    print(f"Loading models from: {args.models_file}")
    all_models = load_models(args.models_file)

    # Filter models if specific ones requested
    if args.models:
        models_to_process = [m for m in all_models if m['name'] in args.models]
        if not models_to_process:
            print(f"Error: No matching models found for: {args.models}")
            return
    else:
        models_to_process = all_models

    print(f"\n{'='*80}")
    print(f"BATCH GENERATION CONFIGURATION")
    print(f"{'='*80}")
    print(f"Models to process: {len(models_to_process)}")
    print(f"Sequences per model: {args.num_sequences}")
    print(f"Max length: {args.max_length}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Sampling seed: None (diverse generations)")
    print(f"Unique prompts: {args.num_unique_prompts or 'auto'}")
    print(f"Prompt seed: {args.prompt_seed}")
    print(f"Prompts file: {args.prompts_file or 'None (using diverse topic-based prompts)'}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nModels:")
    for i, model in enumerate(models_to_process, 1):
        model_name = model['name']
        model_size = model.get('size', '?')
        print(f"  {i}. {model_name} ({model_size})")
    print()

    # Confirm before starting
    confirm = input("Proceed with generation? [y/N]: ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    # Process each model
    start_time = datetime.now()
    results = {}

    for i, model in enumerate(models_to_process, 1):
        model_name = model['name']
        print(f"\n{'#'*80}")
        print(f"# MODEL {i}/{len(models_to_process)}")
        print(f"{'#'*80}")

        success = run_generation(
            model_name=model_name,
            num_sequences=args.num_sequences,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            num_unique_prompts=args.num_unique_prompts,
            prompt_seed=args.prompt_seed,
            prompts_file=args.prompts_file,
            output_dir=args.output_dir
        )

        results[model_name] = "✓ Success" if success else "✗ Failed"

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'='*80}")
    print(f"BATCH GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {duration}")
    print(f"\nResults:")
    for model_name, status in results.items():
        print(f"  {status}: {model_name}")

    successful = sum(1 for s in results.values() if "Success" in s)
    print(f"\nSuccessful: {successful}/{len(results)}")
    print(f"Output directory: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
