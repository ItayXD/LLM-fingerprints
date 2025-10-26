#!/usr/bin/env python3
"""
Batch pairwise knowledge distillation for all combinations of models.
Distills model A onto model B using text generated from model A.
"""

import argparse
import subprocess
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple
from itertools import product


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


def get_model_safe_name(model_name: str) -> str:
    """Convert model name to safe filename."""
    return model_name.replace("/", "_")


def find_or_generate_training_data(
    model_name: str,
    data_dir: str = "distillation_text",
    num_sequences: int = 10000,
    max_length: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.95,
    num_unique_prompts: int = 1000,
    prompt_seed: int = 42
) -> str:
    """Find existing training data or generate it if not found.

    Args:
        model_name: Full model name (e.g., meta-llama/Llama-3.2-1B)
        data_dir: Directory containing training data
        num_sequences: Number of sequences to generate (if needed)
        max_length: Max length per sequence (if generating)
        temperature: Sampling temperature (if generating)
        top_p: Nucleus sampling parameter (if generating)
        num_unique_prompts: Number of unique prompts (if generating)
        prompt_seed: Random seed for prompts (if generating)

    Returns:
        Path to training data file

    Raises:
        RuntimeError: If data generation fails
    """
    # Check both current dir and parent dir
    safe_name = get_model_safe_name(model_name)
    data_file = f"{safe_name}.txt"

    possible_paths = [
        os.path.join(data_dir, data_file),
        os.path.join("..", data_dir, data_file),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"✓ Found existing training data: {path}")
            return path

    # Data doesn't exist, generate it
    print(f"Training data not found for {model_name}, generating...")

    # Find generate_training_data.py
    if os.path.exists("generate_training_data.py"):
        script_path = "generate_training_data.py"
    elif os.path.exists("code/generate_training_data.py"):
        script_path = "code/generate_training_data.py"
    else:
        raise RuntimeError("Cannot find generate_training_data.py")

    # Ensure output directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Build command
    cmd = [
        "python", script_path,
        "--model", model_name,
        "--num-sequences", str(num_sequences),
        "--max-length", str(max_length),
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--num-unique-prompts", str(num_unique_prompts),
        "--prompt-seed", str(prompt_seed),
        "--output-dir", data_dir,
    ]

    print(f"\nGenerating {num_sequences} sequences...")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            # Return path to generated file
            generated_path = os.path.join(data_dir, data_file)
            if os.path.exists(generated_path):
                print(f"✓ Generated training data: {generated_path}")
                return generated_path
            else:
                raise RuntimeError(f"Generation succeeded but file not found: {generated_path}")
        else:
            raise RuntimeError(f"Generation failed with code {result.returncode}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error generating training data: {e}")


def run_distillation(
    teacher_model: str,
    student_model: str,
    data_file: str,
    save_dir: str,
    batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 5e-5,
    temperature: float = 3.0,
    gradient_accumulation_steps: int = 1,
    save_every_n_epochs: int = 2
) -> bool:
    """Run knowledge distillation for a teacher-student pair.

    Returns:
        True if successful, False otherwise
    """
    # Find knowledge_distillation.py
    if os.path.exists("knowledge_distillation.py"):
        script_path = "knowledge_distillation.py"
    elif os.path.exists("code/knowledge_distillation.py"):
        script_path = "code/knowledge_distillation.py"
    else:
        print(f"Error: Cannot find knowledge_distillation.py")
        return False

    # Build command
    cmd = [
        "python", script_path,
        "--teacher-model", teacher_model,
        "--student-model", student_model,
        "--mode", "supervised",
        "--data-file", data_file,
        "--batch-size", str(batch_size),
        "--num-epochs", str(num_epochs),
        "--learning-rate", str(learning_rate),
        "--temperature", str(temperature),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
        "--save-dir", save_dir,
        "--save-every-n-epochs", str(save_every_n_epochs),
    ]

    print(f"\n{'='*80}")
    print(f"DISTILLATION: {get_model_safe_name(teacher_model)} → {get_model_safe_name(student_model)}")
    print(f"{'='*80}")
    print(f"Teacher: {teacher_model}")
    print(f"Student: {student_model}")
    print(f"Data: {data_file}")
    print(f"Output: {save_dir}")
    print(f"\nCommand: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error in distillation: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch pairwise knowledge distillation for model combinations"
    )
    parser.add_argument(
        "--models-file",
        type=str,
        default="models.json",
        help="Path to models JSON file (default: models.json)"
    )
    parser.add_argument(
        "--num-models", "-n",
        type=int,
        default=4,
        help="Number of models to use from the list (default: 4, uses first N models)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="distillation_text",
        help="Directory containing training data (default: distillation_text)"
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default="distilled_models",
        help="Base directory for distilled models (default: distilled_models)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training (default: 4)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=3.0,
        help="Distillation temperature (default: 3.0)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)"
    )
    parser.add_argument(
        "--save-every-n-epochs",
        type=int,
        default=2,
        help="Save checkpoint every N epochs (default: 2)"
    )
    parser.add_argument(
        "--skip-same-model",
        action="store_true",
        default=True,
        help="Skip distilling a model onto itself (default: True)"
    )
    parser.add_argument(
        "--gen-num-sequences",
        type=int,
        default=10000,
        help="Number of sequences to generate if data doesn't exist (default: 10000)"
    )
    parser.add_argument(
        "--gen-max-length",
        type=int,
        default=512,
        help="Max length for generated sequences (default: 512)"
    )
    parser.add_argument(
        "--gen-temperature",
        type=float,
        default=1.0,
        help="Temperature for text generation (default: 1.0)"
    )
    parser.add_argument(
        "--gen-num-unique-prompts",
        type=int,
        default=1000,
        help="Number of unique prompts for generation (default: 1000)"
    )
    parser.add_argument(
        "--gen-prompt-seed",
        type=int,
        default=42,
        help="Prompt seed for generation (default: 42)"
    )

    args = parser.parse_args()

    # Load models
    print(f"Loading models from: {args.models_file}")
    all_models = load_models(args.models_file)

    # Take first N models
    models_to_use = all_models[:args.num_models]

    print(f"\n{'='*80}")
    print(f"PAIRWISE DISTILLATION CONFIGURATION")
    print(f"{'='*80}")
    print(f"Using first {len(models_to_use)} models:")
    for i, model in enumerate(models_to_use, 1):
        print(f"  {i}. {model['name']} ({model.get('size', '?')})")

    # Generate all pairwise combinations
    pairs = []
    for teacher in models_to_use:
        for student in models_to_use:
            teacher_name = teacher['name']
            student_name = student['name']

            # Skip if same model (unless explicitly allowed)
            if args.skip_same_model and teacher_name == student_name:
                continue

            pairs.append((teacher_name, student_name))

    print(f"\nTotal distillation pairs: {len(pairs)}")
    print(f"\nDistillation configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  Save every N epochs: {args.save_every_n_epochs}")
    print(f"\nData generation (if needed):")
    print(f"  Sequences: {args.gen_num_sequences}")
    print(f"  Max length: {args.gen_max_length}")
    print(f"  Temperature: {args.gen_temperature}")
    print(f"  Unique prompts: {args.gen_num_unique_prompts}")
    print(f"  Prompt seed: {args.gen_prompt_seed}")
    print(f"\nPairs to distill:")
    for i, (teacher, student) in enumerate(pairs, 1):
        teacher_short = get_model_safe_name(teacher)
        student_short = get_model_safe_name(student)
        print(f"  {i}. {teacher_short} → {student_short}")
    print()

    # Confirm before starting
    confirm = input("Proceed with distillation? [y/N]: ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    # Run distillations
    start_time = datetime.now()
    results = {}

    for i, (teacher_name, student_name) in enumerate(pairs, 1):
        print(f"\n{'#'*80}")
        print(f"# DISTILLATION {i}/{len(pairs)}")
        print(f"{'#'*80}")

        # Get safe names for paths
        teacher_safe = get_model_safe_name(teacher_name)
        student_safe = get_model_safe_name(student_name)

        # Find or generate training data
        try:
            data_file = find_or_generate_training_data(
                model_name=teacher_name,
                data_dir=args.data_dir,
                num_sequences=args.gen_num_sequences,
                max_length=args.gen_max_length,
                temperature=args.gen_temperature,
                num_unique_prompts=args.gen_num_unique_prompts,
                prompt_seed=args.gen_prompt_seed
            )
        except RuntimeError as e:
            print(f"\n✗ {e}")
            results[f"{teacher_safe}→{student_safe}"] = "✗ Failed (data generation error)"
            continue

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(
            args.output_base_dir,
            f"{teacher_safe}_to_{student_safe}",
            timestamp
        )

        # Run distillation
        success = run_distillation(
            teacher_model=teacher_name,
            student_model=student_name,
            data_file=data_file,
            save_dir=save_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_every_n_epochs=args.save_every_n_epochs
        )

        results[f"{teacher_safe}→{student_safe}"] = "✓ Success" if success else "✗ Failed"

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'='*80}")
    print(f"BATCH DISTILLATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {duration}")
    print(f"\nResults:")
    for pair, status in results.items():
        print(f"  {status}: {pair}")

    successful = sum(1 for s in results.values() if "Success" in s)
    print(f"\nSuccessful: {successful}/{len(results)}")
    print(f"Output directory: {args.output_base_dir}")
    print()


if __name__ == "__main__":
    main()
