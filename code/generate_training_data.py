#!/usr/bin/env python3
"""
Generate training data from a teacher model for knowledge distillation.
Saves generated text to files for reuse in multiple distillation experiments.
"""

import argparse
from vllm import LLM, SamplingParams
import json
import os
import random
from datetime import datetime
from typing import List, Optional


# Diverse topics for generating prompts
TOPICS = [
    "artificial intelligence", "cooking", "travel", "history", "science",
    "technology", "music", "art", "sports", "medicine", "economics",
    "philosophy", "literature", "mathematics", "astronomy", "biology",
    "chemistry", "physics", "psychology", "sociology", "education",
    "environment", "climate", "politics", "business", "architecture",
    "fashion", "photography", "cinema", "theater", "dance", "poetry",
    "mythology", "archaeology", "geology", "oceanography", "meteorology",
    "engineering", "robotics", "space exploration", "quantum mechanics",
    "genetics", "neuroscience", "machine learning", "cryptography",
    "nutrition", "fitness", "meditation", "gardening", "wildlife"
]

# Different types of prompt continuations
PROMPT_TEMPLATES = [
    "Explain the concept of {}",
    "Write a story about {}",
    "Describe the history of {}",
    "What are the benefits of {}?",
    "How does {} work?",
    "The future of {} is",
    "A beginner's guide to {}",
    "The importance of {} in modern society",
    "Interesting facts about {}",
    "The relationship between {} and technology",
    "Common misconceptions about {}",
    "The science behind {}",
    "Why {} matters",
    "The evolution of {}",
    "Innovations in {}",
    "The impact of {} on daily life",
    "Understanding {} from first principles",
    "The art and science of {}",
    "A deep dive into {}",
    "The basics of {}",
    "Advanced techniques in {}",
    "The philosophy of {}",
    "Practical applications of {}",
    "The cultural significance of {}",
    "Recent developments in {}",
    "The challenges of {}",
    "Exploring {} through different perspectives",
    "The ethics of {}",
    "How to get started with {}",
    "The beauty of {}"
]


def generate_diverse_prompts(num_prompts: int, seed: Optional[int] = None) -> List[str]:
    """Generate diverse prompts by combining topics and templates.

    Args:
        num_prompts: Number of unique prompts to generate
        seed: Random seed for reproducibility (None for random)

    Returns:
        List of generated prompt strings
    """
    if seed is not None:
        random.seed(seed)

    prompts = []
    for _ in range(num_prompts):
        topic = random.choice(TOPICS)
        template = random.choice(PROMPT_TEMPLATES)
        prompt = template.format(topic)
        prompts.append(prompt)

    return prompts


def generate_training_data(
    model_name: str,
    num_sequences: int = 1000,
    max_length: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.95,
    prompts: Optional[List[str]] = None,
    num_unique_prompts: Optional[int] = None,
    use_diverse_prompts: bool = True,
    prompt_seed: Optional[int] = None,
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
        prompts: Optional list of prompts to start from
        num_unique_prompts: Number of unique prompts (will cycle to reach num_sequences)
        use_diverse_prompts: Whether to use generated diverse prompts
        prompt_seed: Random seed for prompt generation (None = random prompts)
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

    print(f"Generating {num_sequences} sequences...")
    print(f"  Max length: {max_length}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-p: {top_p}")
    print(f"  Sampling seed: None (different generations for same prompt)")

    # Prepare prompts
    if prompts and len(prompts) > 0:
        # Use provided prompts
        print(f"  Using {len(prompts)} provided prompts")
        unique_prompts = prompts
    elif use_diverse_prompts:
        # Generate diverse prompts from topics and templates
        n_unique = num_unique_prompts or min(1000, num_sequences)
        print(f"  Generating {n_unique} diverse prompts from topics and templates")
        print(f"  Prompt generation seed: {prompt_seed if prompt_seed else 'random'}")
        unique_prompts = generate_diverse_prompts(n_unique, seed=prompt_seed)
    else:
        # Fallback to minimal prompt
        print(f"  Using minimal prompt (unconditional)")
        unique_prompts = ["\n"]

    # Cycle through prompts to generate num_sequences
    # This allows multiple different generations from the same prompt
    input_prompts = [unique_prompts[i % len(unique_prompts)] for i in range(num_sequences)]

    if len(unique_prompts) < num_sequences:
        print(f"  Cycling {len(unique_prompts)} prompts to generate {num_sequences} sequences")
        print(f"  Each prompt will generate ~{num_sequences // len(unique_prompts)} variations")

    print()

    # Configure sampling parameters - NO SEED for varied generations
    sampling_params = SamplingParams(
        n=1,
        max_tokens=max_length,
        temperature=temperature,
        top_p=top_p,
        seed=None,  # No seed = different generations for same prompt
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
        "--prompts-file",
        type=str,
        default=None,
        help="JSON file with prompts to use as starting points (uses 'prompts' key)"
    )
    parser.add_argument(
        "--num-unique-prompts",
        type=int,
        default=None,
        help="Number of unique prompts to generate (cycles to reach num-sequences, default: min(1000, num_sequences))"
    )
    parser.add_argument(
        "--prompt-seed",
        type=int,
        default=None,
        help="Random seed for prompt generation (default: None for random prompts)"
    )
    parser.add_argument(
        "--no-diverse-prompts",
        action="store_true",
        help="Disable diverse prompt generation (use minimal prompt instead)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="distillation_text",
        help="Output directory (default: distillation_text)"
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
        prompts=prompts,
        num_unique_prompts=args.num_unique_prompts,
        use_diverse_prompts=not args.no_diverse_prompts,
        prompt_seed=args.prompt_seed,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype
    )

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Determine output filename (no timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Keep for metadata
    if args.output_name:
        output_name = args.output_name
    else:
        model_safe_name = args.model.replace("/", "_")
        output_name = model_safe_name

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
        "sampling_seed": None,  # Always None for diverse generations
        "prompts_file": args.prompts_file,
        "use_diverse_prompts": not args.no_diverse_prompts,
        "num_unique_prompts": args.num_unique_prompts,
        "prompt_seed": args.prompt_seed,
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
