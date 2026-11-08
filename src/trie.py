"""
Trie for fast prefix and exact-match lookups against a Polish word dictionary.

Also stores the top-N most frequent words from the CSV for use as a last-resort
fallback when beam search finds no valid predictions.
"""

import csv
from typing import Dict, List, Optional, Tuple


class TrieNode:
    __slots__ = ("children", "is_word")

    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.is_word: bool = False


class PolishWordTrie:
    """
    Character-level trie over a Polish word dictionary.

    All words are stored lowercased. Lookups are O(len(word)) regardless
    of dictionary size.

    Also exposes top_n_words — the N most frequent words read from the CSV,
    used as a last-resort fallback when beam search produces no results.
    """

    def __init__(self):
        self._root = TrieNode()
        self._word_count = 0
        self.top_n_words: List[str] = []   # populated by from_csv()

    def insert(self, word: str) -> None:
        word = word.strip().lower()
        if not word:
            return
        node = self._root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        if not node.is_word:
            node.is_word = True
            self._word_count += 1

    def is_valid_prefix(self, prefix: str) -> bool:
        """True if any dictionary word starts with prefix. O(len(prefix))."""
        prefix = prefix.strip().lower()
        node = self._root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def is_word(self, word: str) -> bool:
        """True if word is an exact match in the dictionary. O(len(word))."""
        word = word.strip().lower()
        node = self._root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_word

    @property
    def word_count(self) -> int:
        return self._word_count

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        word_column: int = 0,
        frequency_column: Optional[int] = 1,
        has_header: bool = True,
        top_n_fallback: int = 100,
    ) -> "PolishWordTrie":
        """
        Build a trie from a CSV file and store the top-N most frequent words.

        Args:
            csv_path         : path to the CSV file
            word_column      : column index containing the words (default: 0)
            frequency_column : column index containing frequencies (default: 1).
                               If None, words are assumed to be sorted by
                               frequency already (first rows = most frequent).
            has_header       : whether the first row is a header (default: True)
            top_n_fallback   : how many top-frequency words to keep as fallback
        """
        trie = cls()
        rows: List[Tuple[str, float]] = []   # (word, frequency)

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            if has_header:
                next(reader, None)
            for row in reader:
                if not row or len(row) <= word_column:
                    continue
                word = row[word_column].strip().lower()
                if not word:
                    continue
                trie.insert(word)

                # Collect frequency for fallback ranking
                if frequency_column is not None and len(row) > frequency_column:
                    try:
                        freq = float(row[frequency_column])
                    except ValueError:
                        freq = 0.0
                else:
                    # No frequency column — use insertion order as a proxy
                    # (assumes CSV is already sorted by descending frequency)
                    freq = -trie._word_count   # decreasing → earlier = higher rank
                rows.append((word, freq))

        # Sort by frequency descending and keep the top-N
        rows.sort(key=lambda x: x[1], reverse=(frequency_column is not None))
        trie.top_n_words = [word for word, _ in rows[:top_n_fallback]]

        print(f"Trie built: {trie.word_count:,} words loaded from {csv_path}")
        print(f"Fallback list: top {len(trie.top_n_words)} most frequent words stored")
        return trie

    @classmethod
    def from_word_list(cls, words) -> "PolishWordTrie":
        """Build a trie from any iterable of strings (no frequency data)."""
        trie = cls()
        for word in words:
            trie.insert(word)
        return trie