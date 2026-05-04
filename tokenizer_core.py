"""
tokenizer_core.py
-----------------
Custom Byte Pair Encoding (BPE) tokenizer extracted from main.ipynb.
Implements get_stats, merge, and BasicTokenizer with train/encode/decode.
"""

import json
import os

# ---------------------------------------------------------------------------
# Core BPE utilities
# ---------------------------------------------------------------------------

def get_stats(ids, counts=None):
    """Count consecutive pair frequencies in the ids list."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """Replace all consecutive occurrences of pair in ids with idx."""
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


# ---------------------------------------------------------------------------
# BasicTokenizer class
# ---------------------------------------------------------------------------

class BasicTokenizer:
    def __init__(self):
        self.merges: dict = {}          # (int, int) -> int
        self.vocab: dict = {}           # int -> bytes
        self.trained: bool = False
        self._init_vocab()

    def _init_vocab(self):
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train(self, text: str, vocab_size: int = 512) -> None:
        """
        Train BPE on text.  Starts from raw UTF-8 bytes (256 tokens) and
        greedily merges the most-frequent pair until vocab_size is reached.
        """
        assert vocab_size >= 256, "vocab_size must be >= 256"
        num_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))
        merges: dict = {}
        vocab: dict = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            print(f"  merge {i + 1}/{num_merges}: {pair} -> {idx}  "
                  f"(freq={stats[pair]})")

        self.merges = merges
        self.vocab = vocab
        self.trained = True

    # -----------------------------------------------------------------------
    # Encode / Decode
    # -----------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Convert a string into a list of integer token IDs."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            # pick the pair that was merged earliest (lowest idx = highest priority)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Convert a list of token IDs back to a string."""
        text_bytes = b"".join(self.vocab.get(idx, b"\xef\xbf\xbd") for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")

    def encode_with_segments(self, text: str):
        """
        Returns (ids, segments) where segments is a list of dicts:
            { "id": int, "text": str, "bytes": list[int] }
        Useful for the frontend colored-token visualisation.
        """
        ids = self.encode(text)
        segments = []
        for token_id in ids:
            token_bytes = self.vocab.get(token_id, b"\xef\xbf\xbd")
            token_text = token_bytes.decode("utf-8", errors="replace")
            segments.append({
                "id": token_id,
                "text": token_text,
                "bytes": list(token_bytes),
            })
        return ids, segments

    # -----------------------------------------------------------------------
    # Persist / load
    # -----------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        data = {
            "merges": [[p0, p1, idx] for (p0, p1), idx in self.merges.items()]
        }
        with open(filepath, "w") as f:
            json.dump(data, f)
        print(f"Tokenizer saved to {filepath}")

    def load(self, filepath: str) -> None:
        with open(filepath, "r") as f:
            data = json.load(f)
        self.merges = {(item[0], item[1]): item[2] for item in data["merges"]}
        self.vocab = self._build_vocab()
        self.trained = True
        print(f"Tokenizer loaded from {filepath} "
              f"({len(self.merges)} merges, vocab_size={256 + len(self.merges)})")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


# ---------------------------------------------------------------------------
# Training corpus  (same Unicode essay used in the notebook)
# ---------------------------------------------------------------------------

TRAINING_TEXT = (
    "A Programmer's Introduction to Unicode March 3, 2017 · Coding · 22 Comments  "
    "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺\u200c🇳\u200c🇮\u200c🇨\u200c🇴\u200c🇩\u200c🇪! 😄 "
    "The very name strikes fear and awe into the hearts of programmers worldwide. "
    "We all know we ought to \u201csupport Unicode\u201d in our software (whatever that means\u2014"
    "like using wchar_t for all the strings, right?). But Unicode can be abstruse, "
    "and diving into the thousand-page Unicode Standard plus its dozens of supplementary "
    "annexes, reports, and notes can be more than a little intimidating. "
    "I don\u2019t blame programmers for still finding the whole thing mysterious, "
    "even 30 years after Unicode\u2019s inception.  A few months ago, I got interested in "
    "Unicode and decided to spend some time learning more about it in detail. "
    "In this article, I\u2019ll give an introduction to it from a programmer\u2019s point of view.  "
    "I\u2019m going to focus on the character set and what\u2019s involved in working with strings "
    "and files of Unicode text. However, in this article I\u2019m not going to talk about "
    "fonts, text layout/shaping/rendering, or localization in detail\u2014those are separate "
    "issues, beyond my scope (and knowledge) here.  The Unicode Codespace consists of "
    "1,114,112 code points. However, only 128,237 of them\u2014about 12% of the codespace\u2014"
    "are actually assigned, to date. There\u2019s plenty of room for growth! "
    "Unicode also reserves an additional 137,468 code points as \u201cprivate use\u201d areas, "
    "which have no standardized meaning and are available for individual applications "
    "to define for their own purposes. UTF-8, in UTF-8, each code point is stored using "
    "1 to 4 bytes, based on its index value. UTF-8 uses a system of binary prefixes. "
    "The most convenient, computer-friendliest thing to do would be to just store the "
    "code point index as a 32-bit integer. This works, but it consumes 4 bytes per code "
    "point, which is sort of a lot. Consequently, there are several more-compact encodings "
    "for Unicode. Language models tokenize text before processing it. Tokenization is the "
    "process of splitting text into tokens. Tokens are the basic units that language models "
    "work with. Understanding tokenization is crucial for working with language models. "
    "The tokenizer converts text into a sequence of integers, each representing a token. "
    "Byte Pair Encoding is a popular tokenization algorithm used by many language models. "
    "It starts with individual bytes and iteratively merges the most frequent pairs. "
    "Python is a popular programming language for machine learning and natural language "
    "processing. The quick brown fox jumps over the lazy dog. "
    "Hello world! This is a tokenizer demonstration. "
    "Artificial intelligence and machine learning are transforming the world. "
    "Natural language processing enables computers to understand human language. "
    "Deep learning neural networks have revolutionized many fields of computer science."
)


# ---------------------------------------------------------------------------
# Singleton: load or train once
# ---------------------------------------------------------------------------

_TOKENIZER_INSTANCE = None
TOKENIZER_SAVE_PATH = os.path.join(os.path.dirname(__file__), "tokenizer_model.json")


def get_tokenizer(vocab_size: int = 512) -> BasicTokenizer:
    """Return a singleton tokenizer, loading from disk or training fresh."""
    global _TOKENIZER_INSTANCE
    if _TOKENIZER_INSTANCE is not None:
        return _TOKENIZER_INSTANCE

    tok = BasicTokenizer()
    if os.path.exists(TOKENIZER_SAVE_PATH):
        tok.load(TOKENIZER_SAVE_PATH)
    else:
        print(f"Training BPE tokenizer (vocab_size={vocab_size}) ...")
        tok.train(TRAINING_TEXT, vocab_size=vocab_size)
        tok.save(TOKENIZER_SAVE_PATH)

    _TOKENIZER_INSTANCE = tok
    return tok


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tok = get_tokenizer(vocab_size=512)
    sample = "Hello, world! This is the tokenizer."
    ids, segs = tok.encode_with_segments(sample)
    print(f"\nSample: {sample!r}")
    print(f"Token count : {len(ids)}")
    print(f"Token IDs   : {ids}")
    print(f"Decoded     : {tok.decode(ids)!r}")
    print("\nSegments:")
    for s in segs:
        print(f"  [{s['id']:>4}] {s['text']!r}")
