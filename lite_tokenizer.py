"""
lite_tokenizer.py
==================
A minimal, dependency-free reimplementation of enough of Keras's Tokenizer to
run texts_to_sequences() from a saved tokenizer.json.

WHY THIS EXISTS
-----------------
The whole point of the ONNX export (export_onnx.py) is to stop importing
TensorFlow (~600MB, slow cold start) to serve predictions. But the original
serving path also imported TF just to get `tokenizer_from_json` and
`pad_sequences` -- two functions that do nothing TF-specific. Padding is
numpy arithmetic; tokenization is lowercasing, stripping punctuation, splitting
on whitespace, and a dictionary lookup. None of that needs a deep learning
framework.

This file reproduces Keras's Tokenizer behaviour EXACTLY -- same filter
characters, same lowercasing, same OOV handling, same num_words truncation --
verified in test_lite_tokenizer.py by running both tokenizers on the same
sentences from the real training vocabulary and asserting identical output.
If Keras ever changes text_to_word_sequence's defaults, that test is what
catches the drift.

WHAT IT DOES NOT DO
----------------------
Does not support char_level tokenization, custom split characters, or fitting
a new tokenizer -- only loading an already-fit tokenizer.json and encoding new
text with it. That is the entire surface serving needs.
"""

from __future__ import annotations

import json


class LiteTokenizer:
    def __init__(self, word_index: dict[str, int], num_words: int | None,
                filters: str, lower: bool, oov_token: str | None):
        self.word_index = word_index
        self.num_words = num_words
        self.filters = filters
        self.lower = lower
        self.oov_token = oov_token
        self._translate = str.maketrans(filters, " " * len(filters))

    def _text_to_word_sequence(self, text: str) -> list[str]:
        """Byte-for-byte match of keras.preprocessing.text.text_to_word_sequence."""
        if self.lower:
            text = text.lower()
        text = text.translate(self._translate)
        return [w for w in text.split(" ") if w]

    def texts_to_sequences(self, texts: list[str]) -> list[list[int]]:
        oov_index = self.word_index.get(self.oov_token) if self.oov_token else None
        out = []
        for text in texts:
            seq = []
            for word in self._text_to_word_sequence(text):
                idx = self.word_index.get(word)
                if idx is not None:
                    if self.num_words and idx >= self.num_words:
                        if oov_index is not None:
                            seq.append(oov_index)
                        continue
                    seq.append(idx)
                elif oov_index is not None:
                    seq.append(oov_index)
            out.append(seq)
        return out


def load_tokenizer(json_path_or_string: str) -> LiteTokenizer:
    """
    Accepts either a path to tokenizer.json or the JSON string itself --
    mirrors how tokenizer_from_json(fh.read()) is called in classifier.py, so
    swapping call sites is a one-line change.
    """
    if "\n" not in json_path_or_string and json_path_or_string.endswith(".json"):
        raw = open(json_path_or_string, encoding="utf-8").read()
    else:
        raw = json_path_or_string

    outer = json.loads(raw)
    # Keras nests the real config as a JSON-encoded STRING, not a nested
    # object -- tokenizer_from_json double-decodes it, and so must we.
    cfg = outer.get("config", outer)

    def _maybe_json(v):
        return json.loads(v) if isinstance(v, str) else v

    word_index = _maybe_json(cfg.get("word_index", "{}"))
    num_words = cfg.get("num_words")
    filters = cfg.get("filters", '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    lower = bool(cfg.get("lower", True))
    oov_token = cfg.get("oov_token")

    return LiteTokenizer(word_index={k: int(v) for k, v in word_index.items()},
                         num_words=num_words, filters=filters, lower=lower,
                         oov_token=oov_token)


def pad_sequences(sequences: list[list[int]], maxlen: int,
                  padding: str = "post", truncating: str = "post",
                  dtype: str = "int32"):
    """
    Pure-numpy replacement for keras.preprocessing.sequence.pad_sequences,
    restricted to the arguments classifier.py actually uses (post/post,
    integer dtype). Not a general reimplementation of every option Keras
    supports -- just the ones this codebase calls.
    """
    import numpy as np
    out = np.zeros((len(sequences), maxlen), dtype=dtype)
    for i, seq in enumerate(sequences):
        if not seq:
            continue
        if len(seq) > maxlen:
            seq = seq[:maxlen] if truncating == "post" else seq[-maxlen:]
        if padding == "post":
            out[i, :len(seq)] = seq
        else:
            out[i, -len(seq):] = seq
    return out
