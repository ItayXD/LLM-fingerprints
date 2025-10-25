"""Simple HuggingFace model utilities for text generation."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List, Dict, Any


class HFModel:
    """Simple wrapper for HuggingFace models."""

    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        """Initialize model and tokenizer.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.device = device if device else self._get_device()

        # Provide info about device
        device_info = {
            'mps': 'Apple Silicon GPU',
            'cuda': 'NVIDIA GPU',
            'cpu': 'CPU'
        }
        print(
            f"Loading {model_name} on {self.device} ({device_info.get(self.device, 'Unknown')})..."
        )

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # For MPS, load in float16 for better performance and memory usage
        if self.device == 'mps':
            print("Using float16 for MPS device (Apple Silicon)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Set pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

    def _get_device(self):
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def generate(self,
                 prompt: str,
                 max_new_tokens: int = 100,
                 min_new_tokens: int = None,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 do_sample: bool = True,
                 seed: Optional[int] = None) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            min_new_tokens: Minimum tokens to generate (optional)
            temperature: Sampling temperature (higher = more random, 0 = greedy)
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample or use greedy decoding
            seed: Random seed for reproducibility
        
        Returns:
            Generated text (without the prompt)
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Handle temperature = 0 case (greedy decoding)
        if temperature == 0 or temperature == 0.0:
            # Use greedy decoding (deterministic)
            do_sample = False
            generate_params = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": self.tokenizer.pad_token_id
            }
        else:
            # Use sampling with temperature
            generate_params = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id
            }

        # Add min_new_tokens if specified
        if min_new_tokens is not None:
            generate_params["min_new_tokens"] = min_new_tokens

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_params)

        # Decode
        generated_text = self.tokenizer.decode(outputs[0],
                                               skip_special_tokens=True)

        # Remove prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        return generated_text

    def get_next_token_probs(self,
                             prompt: str,
                             top_k: int = 10) -> List[tuple]:
        """Get probabilities for next token.
        
        Args:
            prompt: Input text
            top_k: Number of top tokens to return
        
        Returns:
            List of (token, probability) tuples
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Get top-k tokens
        top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

        results = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            token = self.tokenizer.decode([idx])
            results.append((token, prob))

        return results

    def generate_with_token_probs(self,
                                  prompt: str,
                                  max_new_tokens: int = 100,
                                  temperature: float = 1.0) -> Dict[str, Any]:
        """Generate text and return token probabilities.
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Dictionary with 'text', 'tokens', and 'probs'
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate with scores
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id)

        # Extract generated tokens
        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_tokens = [
            self.tokenizer.decode([tid]) for tid in generated_ids
        ]
        generated_text = self.tokenizer.decode(generated_ids,
                                               skip_special_tokens=True)

        # Extract probabilities
        token_probs = []
        if outputs.scores:
            for i, score in enumerate(outputs.scores):
                probs = torch.softmax(score[0] / temperature, dim=-1)
                token_id = generated_ids[i] if i < len(
                    generated_ids) else self.tokenizer.eos_token_id
                token_prob = probs[token_id].item()
                token_probs.append(token_prob)

        return {
            'text': generated_text,
            'tokens': generated_tokens,
            'probs': token_probs
        }


# Convenience functions for quick usage
def quick_generate(prompt: str,
                   model_name: str = "gpt2",
                   max_new_tokens: int = 100,
                   temperature: float = 1.0) -> str:
    """Quick generation without keeping model in memory.
    
    Args:
        prompt: Input text
        model_name: HuggingFace model name
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated text
    """
    model = HFModel(model_name)
    return model.generate(prompt, max_new_tokens, temperature)


# Example usage
if __name__ == "__main__":
    # Method 1: Quick one-off generation
    text = quick_generate("The capital of France is",
                          model_name="gpt2",
                          max_new_tokens=20)
    print(f"Quick generate: {text}\n")

    # Method 2: Reuse model for multiple generations
    model = HFModel("gpt2")

    # Simple generation
    text = model.generate("Once upon a time",
                          max_new_tokens=30,
                          temperature=0.8)
    print(f"Generated story: {text}\n")

    # Get next token probabilities
    next_tokens = model.get_next_token_probs("The weather today is", top_k=5)
    print("Next token predictions:")
    for token, prob in next_tokens:
        print(f"  '{token}': {prob:.3f}")

    # Generate with token probabilities
    result = model.generate_with_token_probs("Python is a",
                                             max_new_tokens=10,
                                             temperature=1.0)
    print(f"\nGenerated: {result['text']}")
    print("Token probabilities:")
    for token, prob in zip(result['tokens'], result['probs']):
        print(f"  '{token}': {prob:.3f}")
