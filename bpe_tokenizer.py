from collections import defaultdict, Counter
import re
from typing import List
from IPython import embed


class BPETokenizer:

    def __init__(self, corpus: List[str], vocab_size: int = 1e3) -> None:
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.train()

    def tokenize(self, text):
        words = [word for word in text.split()]
        splits = [[l for l in word] for word in words]
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                splits[idx] = split

        return sum(splits, [])

    def train(self):
        word_freqs = defaultdict(int)

        for text in self.corpus:
            for word in [word for word in text.split()]:
                word_freqs[word] += 1

        alphabet = []

        for word in word_freqs.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.append(letter)

        alphabet.sort()

        vocab = ["<|endoftext|>"] + alphabet.copy()

        splits = {word: [c for c in word] for word in word_freqs.keys()}

        def compute_pair_freqs(splits):
            pair_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                split = splits[word]
                if len(split) == 1:
                    continue
                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    pair_freqs[pair] += freq
            return pair_freqs

        def merge_pair(a, b, splits):
            for word in word_freqs:
                split = splits[word]
                if len(split) == 1:
                    continue

                i = 0
                while i < len(split) - 1:
                    if split[i] == a and split[i + 1] == b:
                        split = split[:i] + [a + b] + split[i + 2:]
                    else:
                        i += 1
                splits[word] = split
            return splits

        self.merges = {}

        while len(vocab) < self.vocab_size:
            print(len(vocab))
            try:
                pair_freqs = compute_pair_freqs(splits)
                best_pair = ""
                max_freq = None
                for pair, freq in pair_freqs.items():
                    if max_freq is None or max_freq < freq:
                        best_pair = pair
                        max_freq = freq
                if best_pair == "":
                    break
                splits = merge_pair(*best_pair, splits)
                self.merges[best_pair] = best_pair[0] + best_pair[1]
                vocab.append(best_pair[0] + best_pair[1])
            except Exception as e:
                embed()


if __name__ == "__main__":
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]

    tokenizer = BPETokenizer(
        corpus=corpus
    )
    print(tokenizer.tokenize("This is not a token."))
