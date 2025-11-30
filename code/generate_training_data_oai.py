#!/usr/bin/env python3
"""
Generate training data from a teacher model for knowledge distillation.
Saves generated text to files for reuse in multiple distillation experiments.
"""

import argparse
import json
import os
import random
from datetime import datetime
from typing import List, Optional
from openai import OpenAI, RateLimitError
from tqdm import tqdm
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

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
    "nutrition", "fitness", "meditation", "gardening", "wildlife", "linguistics",
    "anthropology", "blockchain", "cybersecurity", "game design", "virtual reality", "augmented reality", "nanotechnology", "biotechnology", "renewable energy", "sustainable agriculture", "urban planning", "ecology", "marine biology", "astrophysics", "AI safety", "epidemiology", "pharmacology", "forensic science", "cognitive science", "ethics", "AI alignment", "social work", "public health", "international relations", "law", "criminology", "human geography", "cartography", "animation", "sound engineering", "graphic design", "industrial design", "web development", "software engineering", "data science", "statistics", "microbiology", "zoology", "botany", "paleontology", "hydrology", "energy policy", "supply chain management", "entrepreneurship", "marketing", "communication studies", "public speaking", "human–computer interaction","computational linguistics",
    "bioinformatics", "systems biology", "tissue engineering", "materials science", "quantitative finance", "behavioral economics", "neuroeconomics", "artificial life", "exoplanet research", "planetary science", "stellar evolution", "geophysics", "volcanology", "seismology", "glaciology", "conservation biology", "environmental engineering", "forestry", "agroecology", "horticulture", "urban sociology", "cultural studies", "media studies", "gender studies", "queer theory", "critical race theory", "semiotics", "rhetoric", "folklore studies", "music theory", "ethnomusicology", "choreography", "set design", "creative writing", "screenwriting", "storytelling", "string theory", "neurology", "ornithology", "herpetology", "biomechanics", "computational physics", "theoretical biology", "information theory", "complex systems", "emergent behavior", "distributed systems"
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


def generate_diverse_prompts(seed: Optional[int] = 42) -> List[str]:
    """Generate diverse prompts by combining topics and templates.

    Args:
        num_prompts: Number of unique prompts to generate
        seed: Random seed for reproducibility (None for random)

    Returns:
        List of generated prompt strings
    """
    if seed is not None:
        random.seed(seed)
        
    total_prompts = len(TOPICS) * len(PROMPT_TEMPLATES)
    
    prompts = []
    for i in range(len(TOPICS)):
        for j in range(len(PROMPT_TEMPLATES)):
            topic = TOPICS[i]
            template = PROMPT_TEMPLATES[j]
            prompt = template.format(topic)
            prompts.append(prompt)

    return prompts


def generate_training_data(
    model_name: str,
    min_num_sequences: int = 1000,
    max_length: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.95,
    prompts: Optional[List[str]] = None, 
    num_unique_prompts: Optional[int] = None,
    use_diverse_prompts: bool = True,
    prompt_seed: Optional[int] = None,
    api_key: Optional[str] = None,
    delay_between_calls: float = 0.0
) -> List[dict]:
    """Generate text sequences from OpenAI models using OpenAI API.

    Args:
        model_name: OpenAI model name (e.g., 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo')
        min_num_sequences: Number of sequences to generate
        max_length: Maximum length per sequence (max tokens)
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        prompts: Optional list of prompts to start from
        num_unique_prompts: Number of unique prompts (will cycle to reach min_num_sequences)
        use_diverse_prompts: Whether to use generated diverse prompts
        prompt_seed: Random seed for prompt generation (None = random prompts)
        api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
        delay_between_calls: Delay in seconds between API calls to avoid rate limits

    Returns:
        List of dicts with format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    # Initialize OpenAI client
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set and no api_key provided")
    
    client = OpenAI(api_key=api_key)
    
    print(f"Using OpenAI API with model: {model_name}")
    print(f"  Max tokens per sequence: {max_length}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-p: {top_p}")
    print(f"  Delay between calls: {delay_between_calls}s")

    # Prepare prompts
    if prompts and len(prompts) > 0:
        # Use provided prompts
        print(f"  Using {len(prompts)} provided prompts")
        unique_prompts = prompts
    elif use_diverse_prompts:
        # Generate diverse prompts from topics and templates
        unique_prompts = generate_diverse_prompts(seed=prompt_seed)
    else:
        # Fallback to minimal prompt
        print(f"  Using minimal prompt (unconditional)")
        unique_prompts = ["Explain the concept of artificial intelligence"]

    # Cycle through prompts to generate min_num_sequences
    # This allows multiple different generations from the same prompt
    input_prompts = unique_prompts #[unique_prompts[i % len(unique_prompts)] for i in range(min_num_sequences)]

    if len(unique_prompts) < min_num_sequences:
        print(f"  Cycling {len(unique_prompts)} prompts to generate {min_num_sequences} sequences")
        print(f"  Each prompt will generate ~{min_num_sequences // len(unique_prompts)} variations")

    print()

    # Generate texts using OpenAI API
    print(f"Generating {len(input_prompts)} sequences with OpenAI API...")
    generated_data = []
    
    for i, prompt in enumerate(tqdm(input_prompts, desc="Generating")):
        try:
            # Call OpenAI API
            response = client.chat.completions.create(
                model=model_name,
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Extract the generated text
            generated_text = response.choices[0].message.content
            
            # Store in conversation format
            conversation = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": generated_text}
                ]
            }
            generated_data.append(conversation)
            
            # Delay between calls to respect rate limits
            if i < len(input_prompts) - 1:  # Don't delay after the last call
                time.sleep(delay_between_calls)
                
        except RateLimitError as e:
            print(f"\n⚠️  Rate limit hit at sequence {i+1}. Waiting 60 seconds...")
            time.sleep(60)
            # Retry the same prompt
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    max_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                generated_text = response.choices[0].message.content
                conversation = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": generated_text}
                    ]
                }
                generated_data.append(conversation)
            except Exception as retry_error:
                print(f"\n❌ Failed to generate sequence {i+1} after retry: {retry_error}")
                # Add empty entry to maintain alignment
                conversation = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": ""}
                    ]
                }
                generated_data.append(conversation)
                
        except Exception as e:
            print(f"\n❌ Error generating sequence {i+1}: {e}")
            # Add empty entry to maintain alignment
            conversation = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": ""}
                ]
            }
            generated_data.append(conversation)

    successful = len([d for d in generated_data if d["messages"][1]["content"]])
    print(f"\n✓ Successfully generated {successful} sequences")
    if any(not d["messages"][1]["content"] for d in generated_data):
        failed = len([d for d in generated_data if not d["messages"][1]["content"]])
        print(f"⚠️  {failed} sequences failed")

    return generated_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data from OpenAI models using OpenAI API"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="OpenAI model name (e.g., gpt-4o, gpt-4o-mini, gpt-3.5-turbo, gpt-4-turbo)"
    )
    parser.add_argument(
        "--min-num-sequences", "-n",
        type=int,
        default=1000,
        help="Minimum number of sequences to generate (default: 1000)"
    )
    parser.add_argument(
        "--max-length", "-l",
        type=int,
        default=512,
        help="Maximum tokens per sequence (default: 512)"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)"
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
        "--prompts-jsonl",
        type=str,
        default=None,
        help="JSONL file containing one prompt per line (either a string or object with 'prompt'/'messages')"
    )
    parser.add_argument(
        "--num-unique-prompts",
        type=int,
        default=None,
        help="Number of unique prompts to generate (cycles to reach num-sequences, default: min(1000, min_num_sequences))"
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
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (default: reads from OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between API calls to avoid rate limits (default: 0.0)"
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
    elif args.prompts_jsonl:
        print(f"Loading prompts from JSONL: {args.prompts_jsonl}")
        prompts = []
        with open(args.prompts_jsonl, 'r') as f:
            for line_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                if isinstance(record, dict):
                    if isinstance(record.get("prompt"), str):
                        prompts.append(record["prompt"])
        print(f"Loaded {len(prompts)} prompts")

    # Generate training data
    generated_data = generate_training_data(
        model_name=args.model,
        min_num_sequences=args.min_num_sequences,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        prompts=prompts,
        num_unique_prompts=args.num_unique_prompts,
        use_diverse_prompts=not args.no_diverse_prompts,
        prompt_seed=args.prompt_seed,
        api_key=args.api_key,
        delay_between_calls=args.delay
    )

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Determine output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Keep for metadata
    if args.output_name:
        output_name = args.output_name
    else:
        model_safe_name = args.model.replace("/", "_").replace("-", "_")
        output_name = model_safe_name

    # Save as JSONL file (one conversation per line)
    jsonl_file = os.path.join(output_dir, f"{output_name}.jsonl")
    with open(jsonl_file, 'w') as f:
        for conversation in generated_data:
            # Only save successful generations
            if conversation["messages"][1]["content"]:
                f.write(json.dumps(conversation) + "\n")

    # Save metadata as separate JSON file
    metadata_file = os.path.join(output_dir, f"{output_name}_metadata.json")
    successful_count = len([d for d in generated_data if d["messages"][1]["content"]])
    failed_count = len([d for d in generated_data if not d["messages"][1]["content"]])
    
    output_metadata = {
        "model": args.model,
        "timestamp": timestamp,
        "min_num_sequences": len(generated_data),
        "successful_sequences": successful_count,
        "failed_sequences": failed_count,
        "max_length": args.max_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "prompts_file": args.prompts_file,
        "use_diverse_prompts": not args.no_diverse_prompts,
        "num_unique_prompts": args.num_unique_prompts,
        "prompt_seed": args.prompt_seed,
        "delay_between_calls": args.delay,
    }

    with open(metadata_file, 'w') as f:
        json.dump(output_metadata, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Generated {successful_count} sequences successfully")
    if failed_count > 0:
        print(f"Failed: {failed_count} sequences")
    print(f"Model: {args.model}")
    print(f"\nOutput files:")
    print(f"  JSONL file: {jsonl_file}")
    print(f"  Metadata file: {metadata_file}")
    print(f"\nFormat: Each line contains:")
    print(f'  {{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}')
    print(f"\nTo use for fine-tuning:")
    print(f"  - Use {jsonl_file} as your training data")
    print(f"  - Compatible with OpenAI fine-tuning format")
    print()


if __name__ == "__main__":
    main()