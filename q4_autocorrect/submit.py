"""
Question 4: Probabilistic AutoCorrect for English words

This module builds a minimum-edit-distance spelling corrector using Shakespeare's
text as the vocabulary and frequency model. The implementation supports one-edit
and two-edit candidates, applies cost weights for insert/delete/replace, and
selects the most probable valid candidate.
"""
# Implement by: Alireza Sepehri

import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

ALPHABET = list(string.ascii_lowercase)
CORPUS_FILE = Path("shakespeare.txt")
SUBMISSION_FILE = Path("submission.csv")


def delete_letter(word: str) -> List[str]:
    """Return all strings formed by deleting one character from the word."""
    return [word[:i] + word[i + 1 :] for i in range(len(word))]


def replace_letter(word: str) -> List[str]:
    """Return all strings formed by replacing one character with a lowercase letter."""
    replacements = []
    for i, char in enumerate(word):
        for letter in ALPHABET:
            if letter != char:
                replacements.append(word[:i] + letter + word[i + 1 :])
    return replacements


def insert_letter(word: str) -> List[str]:
    """Return all strings formed by inserting one lowercase letter at any position."""
    inserts = []
    for i in range(len(word) + 1):
        for letter in ALPHABET:
            inserts.append(word[:i] + letter + word[i:])
    return inserts


def edit_one_letter(word: str) -> Set[str]:
    """Return all words that are one edit distance away from the input word."""
    edit_one_set = set()
    edit_one_set.update(delete_letter(word))
    edit_one_set.update(replace_letter(word))
    edit_one_set.update(insert_letter(word))
    return edit_one_set


def edit_two_letters(word: str) -> Set[str]:
    """Return all words that are exactly two edits away from the input word."""
    edit_two_set = set()
    for first_edit in edit_one_letter(word):
        edit_two_set.update(edit_one_letter(first_edit))
    edit_two_set.discard(word)
    return edit_two_set


def _edit_one_letter_with_costs(word: str) -> Dict[str, int]:
    """Return one-edit words with their operation costs."""
    costs: Dict[str, int] = {}
    for candidate in delete_letter(word):
        costs[candidate] = min(costs.get(candidate, float('inf')), 1)
    for candidate in insert_letter(word):
        costs[candidate] = min(costs.get(candidate, float('inf')), 1)
    for candidate in replace_letter(word):
        costs[candidate] = min(costs.get(candidate, float('inf')), 2)
    return costs


def _load_vocabulary(corpus_path: Path) -> Counter:
    """Load words from the Shakespeare corpus and return a frequency counter."""
    text = corpus_path.read_text(encoding="utf-8")
    words = re.findall(r"[a-z]+", text.lower())
    return Counter(words)


def _candidate_costs(word: str, vocabulary: Set[str]) -> Dict[str, int]:
    """Generate candidate corrections and compute minimum edit costs for vocabulary words."""
    costs: Dict[str, int] = {}

    # Direct candidates from one edit
    first_costs = _edit_one_letter_with_costs(word)
    for candidate, cost in first_costs.items():
        if candidate in vocabulary:
            costs[candidate] = min(costs.get(candidate, float('inf')), cost)

    # Two-edit candidates with accumulated costs
    for first_candidate, first_cost in first_costs.items():
        second_costs = _edit_one_letter_with_costs(first_candidate)
        for candidate, second_cost in second_costs.items():
            total_cost = first_cost + second_cost
            if candidate == word:
                continue
            if candidate in vocabulary:
                costs[candidate] = min(costs.get(candidate, float('inf')), total_cost)

    return costs


def autoCorrect(word: str) -> str:
    """Auto-correct a single word based on Shakespeare vocabulary and edit costs."""
    word = word.lower().strip()
    if not word:
        return word

    vocabulary_counts = _load_vocabulary(CORPUS_FILE)
    vocabulary = set(vocabulary_counts)
    total_words = sum(vocabulary_counts.values())

    if word in vocabulary:
        return word

    candidate_costs = _candidate_costs(word, vocabulary)
    if not candidate_costs:
        return word

    best_cost = min(candidate_costs.values())
    best_candidates = [w for w, c in candidate_costs.items() if c == best_cost]

    def score(candidate: str) -> float:
        return vocabulary_counts[candidate] / total_words

    best_candidates.sort(key=lambda w: (-score(w), w))
    return best_candidates[0]


def main() -> None:
    corpus_path = CORPUS_FILE
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {corpus_path}. Please place Shakespeare text at this path."
        )

    vocabulary_counts = _load_vocabulary(corpus_path)
    print(f"Loaded {sum(vocabulary_counts.values())} tokens and {len(vocabulary_counts)} unique words.")
    print("The autoCorrect function is ready to use.")


if __name__ == "__main__":
    main()
