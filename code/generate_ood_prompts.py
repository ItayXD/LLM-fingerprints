"""Generate 1000 out-of-distribution prompts for model fingerprinting."""

import json
import random
import string

random.seed(42)  # For reproducibility


def generate_random_string(min_len=5, max_len=15):
    """Generate random alphanumeric string."""
    length = random.randint(min_len, max_len)
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def generate_leetspeak(words):
    """Convert words to leetspeak."""
    leet_map = {'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5', 't': '7', 'l': '1'}
    result = []
    for word in words:
        leet_word = ''.join(leet_map.get(c, c) for c in word)
        result.append(leet_word)
    return '_'.join(result)


def generate_keyboard_smash():
    """Generate keyboard smash patterns."""
    keyboard_rows = [
        'qwertyuiop',
        'asdfghjkl',
        'zxcvbnm'
    ]
    length = random.randint(6, 12)
    row = random.choice(keyboard_rows)
    return ''.join(random.choices(row, k=length))


def generate_number_letter_mix():
    """Generate mixed number-letter patterns."""
    patterns = []
    for _ in range(random.randint(3, 6)):
        if random.random() > 0.5:
            patterns.append(str(random.randint(0, 999)))
        else:
            patterns.append(''.join(random.choices(string.ascii_lowercase, k=random.randint(2, 5))))
    return ''.join(patterns)


def generate_nonsense_phrase():
    """Generate nonsensical word combinations."""
    adjectives = [
        'purple', 'quantum', 'crystalline', 'ethereal', 'rainbow', 'cosmic',
        'liquid', 'frozen', 'digital', 'phantom', 'neon', 'holographic',
        'magnetic', 'sonic', 'plastic', 'electric', 'mystic', 'atomic',
        'velvet', 'chrome', 'silver', 'golden', 'ruby', 'emerald'
    ]

    nouns = [
        'cat', 'elephant', 'banana', 'computer', 'dream', 'memory',
        'shadow', 'crystal', 'robot', 'dinosaur', 'penguin', 'mushroom',
        'tornado', 'ocean', 'mountain', 'butterfly', 'dragon', 'unicorn',
        'pixel', 'circuit', 'photon', 'nebula', 'vortex', 'matrix'
    ]

    verbs = [
        'dance', 'sing', 'compute', 'dream', 'whisper', 'float',
        'shimmer', 'dissolve', 'transform', 'oscillate', 'radiate', 'pulse',
        'cascade', 'echo', 'resonate', 'vibrate', 'spiral', 'merge'
    ]

    adverbs = [
        'quickly', 'softly', 'mysteriously', 'digitally', 'infinitely',
        'backwards', 'sideways', 'randomly', 'silently', 'slowly',
        'brightly', 'darkly', 'smoothly', 'wildly', 'gently'
    ]

    # Generate different phrase patterns
    patterns = [
        lambda: f"{random.choice(adjectives)} {random.choice(nouns)} {random.choice(verbs)}",
        lambda: f"{random.choice(nouns)} {random.choice(verbs)} {random.choice(adverbs)}",
        lambda: f"{random.choice(adjectives)} {random.choice(adjectives)} {random.choice(nouns)}",
        lambda: f"{random.choice(nouns)} {random.choice(verbs)} {random.choice(adjectives)} {random.choice(nouns)}",
        lambda: f"{random.choice(verbs)} {random.choice(adjectives)} {random.choice(nouns)}",
    ]

    return random.choice(patterns)()


def generate_repeated_pattern():
    """Generate repeated character patterns."""
    chars = random.choice([
        string.ascii_lowercase,
        string.digits,
        string.ascii_lowercase + string.digits
    ])
    pattern_len = random.randint(2, 4)
    repeats = random.randint(2, 5)
    pattern = ''.join(random.choices(chars, k=pattern_len))
    return pattern * repeats


def generate_camelcase_nonsense():
    """Generate CamelCase nonsense words."""
    parts = []
    for _ in range(random.randint(2, 4)):
        length = random.randint(3, 7)
        part = ''.join(random.choices(string.ascii_lowercase, k=length))
        parts.append(part.capitalize())
    return ''.join(parts)


def generate_special_char_mix():
    """Generate strings with special characters."""
    chars = string.ascii_lowercase + string.digits + '_-.'
    length = random.randint(8, 15)
    return ''.join(random.choices(chars, k=length))


def generate_all_caps_gibberish():
    """Generate all caps gibberish."""
    length = random.randint(5, 12)
    return ''.join(random.choices(string.ascii_uppercase, k=length))


def generate_hex_like():
    """Generate hex-like strings."""
    length = random.randint(8, 16)
    return ''.join(random.choices(string.hexdigits.lower(), k=length))


def generate_base64_like():
    """Generate base64-like strings."""
    chars = string.ascii_letters + string.digits + '+/'
    length = random.randint(10, 20)
    result = ''.join(random.choices(chars, k=length))
    # Add some '=' padding randomly
    if random.random() > 0.5:
        result += '=' * random.randint(1, 2)
    return result


def main():
    """Generate 1000 OOD prompts."""
    prompts = []

    # Generate prompts with different strategies
    generators = [
        (generate_random_string, 150),
        (generate_keyboard_smash, 100),
        (generate_number_letter_mix, 100),
        (generate_nonsense_phrase, 200),
        (generate_repeated_pattern, 80),
        (generate_camelcase_nonsense, 80),
        (generate_special_char_mix, 80),
        (generate_all_caps_gibberish, 70),
        (generate_hex_like, 70),
        (generate_base64_like, 70),
    ]

    for generator_func, count in generators:
        for _ in range(count):
            prompts.append(generator_func())

    # Shuffle to mix different types
    random.shuffle(prompts)

    # Ensure we have exactly 1000 unique prompts
    prompts = list(set(prompts))  # Remove duplicates
    while len(prompts) < 1000:
        # Generate more if needed
        generator_func = random.choice([g[0] for g in generators])
        prompts.append(generator_func())
        prompts = list(set(prompts))  # Remove duplicates again

    prompts = prompts[:1000]  # Take exactly 1000

    # Save to JSON
    output = {
        "description": "1000 out-of-distribution prompts for model fingerprinting",
        "count": len(prompts),
        "seed": 42,
        "prompts": prompts
    }

    with open('ood_prompts.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Generated {len(prompts)} OOD prompts")
    print(f"Saved to: ood_prompts.json")

    # Print some examples
    print("\nExample prompts:")
    for i in range(10):
        print(f"  {i+1}. {prompts[i]}")


if __name__ == "__main__":
    main()
