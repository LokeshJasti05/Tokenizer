# Tokenizer

A tokenizer built from scratch in Python.

## Overview

This project implements a tokenizer from the ground up — no external NLP libraries. It takes raw text as input and breaks it down into meaningful units called **tokens**, which is the first step in building a language model, compiler, or any text-processing pipeline.

## What is a Tokenizer?

A tokenizer converts a string of text into a sequence of tokens. Tokens can represent:

- Words or subwords
- Characters
- Punctuation
- Special symbols (e.g., `<PAD>`, `<EOS>`, `<UNK>`)

## Features

- ✅ Built from scratch in pure Python
- ✅ No external NLP dependencies
- ✅ Character-level and/or word-level tokenization
- ✅ Vocabulary builder
- ✅ Encode (text → token IDs) and decode (token IDs → text)

## Getting Started

### Prerequisites

- Python 3.8+

### Installation

```bash
git clone https://github.com/LokeshJasti05/Tokenizer.git
cd Tokenizer
```

### Usage

```python
from tokenizer import Tokenizer

tokenizer = Tokenizer()

# Train on a corpus
tokenizer.train("Hello, world! This is a tokenizer.")

# Encode text to token IDs
ids = tokenizer.encode("Hello world")
print(ids)  # e.g. [4, 7]

# Decode token IDs back to text
text = tokenizer.decode(ids)
print(text)  # "Hello world"
```

## Project Structure

```
Tokenizer/
├── tokenizer.py     # Core tokenizer implementation
├── vocab.py         # Vocabulary management
├── utils.py         # Helper utilities
├── tests/           # Unit tests
└── README.md
```

## How It Works

1. **Preprocessing** — Clean and normalize the raw input text.
2. **Splitting** — Break text into candidate tokens (characters, words, or subwords).
3. **Vocabulary** — Build a mapping from tokens to integer IDs.
4. **Encoding** — Convert input text into a list of integer IDs.
5. **Decoding** — Convert a list of integer IDs back into text.

## Roadmap

- [ ] Basic word-level tokenizer
- [ ] Character-level tokenizer
- [ ] Byte Pair Encoding (BPE)
- [ ] Special token support (`<PAD>`, `<UNK>`, `<EOS>`, `<BOS>`)
- [ ] Save / load vocabulary

## License

This project is licensed under the [MIT License](LICENSE).
