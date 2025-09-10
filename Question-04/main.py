# Spelling Correction using Noisy Channel Model

# Step 1: Create a simple dictionary (V) with at least 5 common words and their prior probabilities
V = {
    "word": 0.4,   # Higher probability for common word
    "world": 0.3,
    "wrote": 0.2,
    "worm": 0.05,
    "worth": 0.05
}

# Step 2: Choose a word from dictionary and create misspelled versions with single-character errors
original_word = "word"
# Example 1: Transposition error: 'o' and 'r' swapped
misspelled_transposition = "wrod"
# Example 2: Substitution error: 'o' -> 'i'
misspelled_substitution = "wird"  # treated as a non-word misspelling in our toy setup

# Step 3: Manually generate candidate set of at least 3 plausible corrections from dictionary
candidates = ["word", "world", "wrote"]

# Helper: check if two strings are exactly one edit apart (insertion, deletion, substitution, or adjacent transposition)
def is_single_edit_apart(a: str, b: str) -> bool:
    if a == b:
        return False  # we expect a non-word misspelling
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False

    # Same length: substitution or adjacent transposition
    if la == lb:
        diffs = [(i, ca, cb) for i, (ca, cb) in enumerate(zip(a, b)) if ca != cb]
        if len(diffs) == 1:
            # single substitution
            return True
        if len(diffs) == 2:
            # possible adjacent transposition
            (i1, a1, b1), (i2, a2, b2) = diffs
            return i2 == i1 + 1 and a1 == b2 and a2 == b1
        return False

    # Length differs by 1: insertion/deletion
    # Ensure a is the shorter one
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    i = j = 0
    edits = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            edits += 1
            if edits > 1:
                return False
            j += 1  # skip one char in longer string (insertion into shorter/deletion from longer)
    # Account for a possible trailing extra char in the longer string
    if j < lb:
        edits += 1
    return edits == 1

# Step 5: Calculate P(s|w) - probability of misspelling given correct word
# Assumption: For single-edit error, P(s|w) = 0.001; otherwise 0.0
def channel_probability(s, w):
    return 0.001 if is_single_edit_apart(s, w) else 0.0

# Utility to compute and print scores for a given misspelling

def evaluate_case(misspelled: str, label: str):
    scores = {}
    for candidate in candidates:
        p_sw = channel_probability(misspelled, candidate)
        p_w = V[candidate]
        scores[candidate] = p_sw * p_w

    best_correction = max(scores, key=scores.get)
    highest_score = scores[best_correction]

    print(f"\n=== {label} ===")
    print(f"Misspelled word: {misspelled}")
    print(f"Candidates: {candidates}")
    print("Scores:")
    for candidate, score in scores.items():
        print(f"  {candidate}: {score}")
    print(f"Best correction: {best_correction} (score: {highest_score})")

# Run both error cases
if __name__ == "__main__":
    evaluate_case(misspelled_transposition, label="Transposition error")
    evaluate_case(misspelled_substitution, label="Substitution error")
