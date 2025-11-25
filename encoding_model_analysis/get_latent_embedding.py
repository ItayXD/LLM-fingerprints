import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# -----------------------
# Configuration
# -----------------------

# Directory containing your per-model JSONL files
INPUT_DIR = Path("/orcd/data/jhm/001/annesyab/LLM/AI_safety/LLM-fingerprints/training_data")

# Directory where we will save embeddings and metadata
OUTPUT_DIR = Path("./encoding_data")

# Name of the sentence-transformers model
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# JSON keys for prompt and response
PROMPT_KEY = "prompt"
RESPONSE_KEY = "response"

# Batch size for encoding
BATCH_SIZE = 64


# -----------------------
# Utilities
# -----------------------

def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Read a .jsonl file into a list of dicts."""
    records = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def ensure_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    

def extract_prompt_response_from_messages(
    record: Dict[str, Any]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a record with a 'messages' list like:
      [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}, ...]
    return (prompt_text, response_text).

    Here we:
      - take the first 'user' message as the prompt
      - take the first 'assistant' message as the response

    Adjust this logic if you want last assistant, multi-turn, etc.
    """
    msgs = record.get("messages", [])
    if not isinstance(msgs, list):
        return None, None

    prompt_text = None
    response_text = None

    for msg in msgs:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user" and prompt_text is None:
            prompt_text = content
        elif role == "assistant" and response_text is None:
            response_text = content

        # Stop early if we already found both
        if prompt_text is not None and response_text is not None:
            break

    return prompt_text, response_text


# -----------------------
# Main encoding function
# -----------------------


def encode_model_responses(
    model_name: str,
    jsonl_path: Path,
    encoder: SentenceTransformer,
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    Read a single model's JSONL file, extract (prompt, response) from 'messages',
    encode all responses, and save embeddings + metadata.

    Saves:
      - embeddings:  <OUTPUT_DIR>/<model_name>_embeddings.npz  (array 'embeddings')
      - metadata:    <OUTPUT_DIR>/<model_name>_metadata.csv    (sample_id, model_name, prompt, response)
    """
    print(f"\nProcessing model: {model_name}")
    records = read_jsonl(jsonl_path)
    print(f"  Loaded {len(records)} records from {jsonl_path}")

    responses = []
    prompts = []
    sample_ids = []

    for idx, rec in enumerate(records):
        prompt_text, resp_text = extract_prompt_response_from_messages(rec)
        if resp_text is None:
            # no assistant response; skip this record
            continue

        responses.append(resp_text)
        prompts.append(prompt_text)
        sample_ids.append(rec.get("id", idx))

    if not responses:
        print("  No valid responses found, skipping.")
        return

    print(f"  Encoding {len(responses)} responses...")

    embeddings_list = []
    for start in tqdm(range(0, len(responses), batch_size)):
        batch = responses[start:start + batch_size]
        emb = encoder.encode(
            batch,
            batch_size=len(batch),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embeddings_list.append(emb)

    embeddings = np.vstack(embeddings_list)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Save embeddings
    emb_out_path = OUTPUT_DIR / f"{model_name}_embeddings.npz"
    np.savez_compressed(emb_out_path, embeddings=embeddings)
    print(f"  Saved embeddings to: {emb_out_path}")

    # Save metadata
    meta = pd.DataFrame({
        "sample_id": sample_ids,
        "model_name": model_name,
        "prompt": prompts,
        "response": responses,
    })
    meta_out_path = OUTPUT_DIR / f"{model_name}_metadata.csv"
    meta.to_csv(meta_out_path, index=False)
    print(f"  Saved metadata to: {meta_out_path}")


# -----------------------
# Driver
# -----------------------

def main():
    ensure_output_dir(OUTPUT_DIR)

    print(f"Loading encoder: {ENCODER_NAME}")
    encoder = SentenceTransformer(ENCODER_NAME)

    # Assume every .jsonl in INPUT_DIR corresponds to one black-box model
    jsonl_files = sorted(INPUT_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {INPUT_DIR}")
        return

    for jsonl_path in jsonl_files:
        # Derive a model name from the file name (without extension)
        model_name = jsonl_path.stem
        encode_model_responses(
            model_name=model_name,
            jsonl_path=jsonl_path,
            encoder=encoder,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
