#!/usr/bin/env python3
"""
Generate training data from a teacher model for knowledge distillation.
Saves generated text to files for reuse in multiple distillation experiments.
"""

import argparse
from vllm import LLM, SamplingParams
import json
import os
from datetime import datetime
from typing import List, Optional


def generate_training_data(
    model_name: str,
    num_sequences: int = 1000,
    max_length: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: Optional[int] = None,
    prompts: Optional[List[str]] = None,
    tensor_parallel_size: int = 1,
    dtype: str = "auto"
) -> List[str]:
    """Generate text sequences from a model using vLLM.

    Args:
        model_name: HuggingFace model name
        num_sequences: Number of sequences to generate
        max_length: Maximum length per sequence
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        seed: Random seed for reproducibility
        prompts: Optional list of prompts to start from (otherwise uses empty string)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        dtype: Data type for model weights

    Returns:
        List of generated text strings
    """
    print(f"Loading model with vLLM: {model_name}")
    print(f"  Tensor parallel size: {tensor_parallel_size}")
    print(f"  Data type: {dtype}")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        trust_remote_code=True
    )

    print("✓ Model loaded successfully!\n")

    # Set random seed if provided
    if seed is not None:
        print(f"Using seed: {seed}")

    print(f"Generating {num_sequences} sequences...")
    print(f"  Max length: {max_length}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-p: {top_p}")

    # Prepare prompts
    if prompts and len(prompts) > 0:
        print(f"  Using {len(prompts)} prompts (cycling if needed)")
        # Cycle through prompts to generate num_sequences
        input_prompts = [prompts[i % len(prompts)] for i in range(num_sequences)]
    else:
        print(f"  Starting from empty string (unconditional)")
        # Use empty string for unconditional generation
        input_prompts = [""] * num_sequences

    print()

    # Configure sampling parameters
    sampling_params = SamplingParams(
        n=1,
        max_tokens=max_length,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )

    # Generate all at once (vLLM batches efficiently)
    print("Running generation (vLLM will batch automatically)...")
    outputs = llm.generate(input_prompts, sampling_params)

    # Extract generated texts
    generated_texts = []
    for output in outputs:
        text = output.outputs[0].text
        generated_texts.append(text)

    print(f"✓ Generated {len(generated_texts)} sequences\n")

    return generated_texts


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data from a teacher model for knowledge distillation"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model name or path (e.g., meta-llama/Llama-3.2-3B)"
    )
    parser.add_argument(
        "--num-sequences", "-n",
        type=int,
        default=1000,
        help="Number of sequences to generate (default: 1000)"
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
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="JSON file with prompts to use as starting points (uses 'prompts' key)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="training_data",
        help="Output directory (default: training_data)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename (default: auto-generated from model name and timestamp)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Data type for model weights (default: auto)"
    )

    args = parser.parse_args()

    # Load prompts if provided
    prompts = None
    if args.prompts_file:
        print(f"Loading prompts from: {args.prompts_file}")
        with open(args.prompts_file, 'r') as f:
            data = json.load(f)
            prompts = data.get('prompts', [])
            print(f"Loaded {len(prompts)} prompts")

    # Generate training data
    generated_texts = generate_training_data(
        model_name=args.model,
        num_sequences=args.num_sequences,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        prompts=prompts,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Determine output filename
    if args.output_name:
        output_name = args.output_name
    else:
        model_safe_name = args.model.replace("/", "_")
        output_name = f"{model_safe_name}"

    # Save as text file (for supervised distillation)
    text_file = os.path.join(output_dir, f"{output_name}.txt")
    with open(text_file, 'w') as f:
        for text in generated_texts:
            f.write(text + "\n")

    # Save as JSON (with metadata)
    json_file = os.path.join(output_dir, f"{output_name}.json")
    output_data = {
        "model": args.model,
        "timestamp": timestamp,
        "num_sequences": len(generated_texts),
        "max_length": args.max_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "prompts_file": args.prompts_file,
        "texts": generated_texts
    }

    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Generated {len(generated_texts)} sequences")
    print(f"Model: {args.model}")
    print(f"\nOutput files:")
    print(f"  Text file: {text_file}")
    print(f"  JSON file: {json_file}")
    print(f"\nTo use for distillation:")
    print(f"  python code/knowledge_distillation.py \\")
    print(f"    --teacher-model {args.model} \\")
    print(f"    --student-model <student-model> \\")
    print(f"    --mode supervised \\")
    print(f"    --data-file {text_file}")
    print()


if __name__ == "__main__":
    main()
