from collections import defaultdict
from typing import Dict, List, Tuple


def build_parallel_corpus() -> Tuple[List[str], List[str]]:
    """Manually constructed English-Malayalam parallel corpus (5 sentences)"""
    english = [
        "I love my country",
        "She is reading a book",
        "The weather is good today",
        "He goes to school",
        "I am learning Malayalam"
    ]

    malayalam = [
        "ഞാൻ എന്റെ രാജ്യത്തെ സ്നേഹിക്കുന്നു",
        "അവൾ ഒരു പുസ്തകം വായിക്കുന്നു",
        "ഇന്ന് കാലാവസ്ഥ നല്ലതാണ്",
        "അവൻ സ്കൂളിലേക്ക് പോകുന്നു",
        "ഞാൻ മലയാളം പഠിക്കുന്നു"
    ]

    return english, malayalam


def tokenize_corpus(sentences: List[str]) -> List[List[str]]:
    """Simple tokenization by splitting on whitespace"""
    return [sent.split() for sent in sentences]


def compute_word_alignment_counts(
    source_corpus: List[List[str]],
    target_corpus: List[List[str]]
) -> Dict[Tuple[str, str], int]:
    """Count co-occurrences of word pairs in aligned sentences"""
    counts = defaultdict(int)

    for src_sent, tgt_sent in zip(source_corpus, target_corpus):
        for src_word in src_sent:
            for tgt_word in tgt_sent:
                counts[(src_word, tgt_word)] += 1

    return counts


def compute_translation_probs(
    word_pair_counts: Dict[Tuple[str, str], int],
    source_vocab: set
) -> Dict[Tuple[str, str], float]:
    """Compute P(target|source) = count(source, target) / sum_over_all_targets(count(source, target))"""
    probs = {}
    source_totals = defaultdict(int)

    for (src, tgt), count in word_pair_counts.items():
        source_totals[src] += count

    for (src, tgt), count in word_pair_counts.items():
        probs[(src, tgt)] = count / source_totals[src]

    return probs


def main():
    print("=" * 80)
    print("TRANSLATION PROBABILITY COMPUTATION: ENGLISH ↔ MALAYALAM")
    print("=" * 80)

    english, malayalam = build_parallel_corpus()

    print("\n[1] Parallel Corpus (5 sentence pairs):")
    print("-" * 80)
    for i, (en, ml) in enumerate(zip(english, malayalam), 1):
        print(f"{i}. EN: {en}")
        print(f"   ML: {ml}")
        print()

    en_tokenized = tokenize_corpus(english)
    ml_tokenized = tokenize_corpus(malayalam)

    print("[2] Tokenized Corpus:")
    print("-" * 80)
    print(f"English tokens: {en_tokenized}")
    print(f"Malayalam tokens: {ml_tokenized}")

    en_vocab = set(word for sent in en_tokenized for word in sent)
    ml_vocab = set(word for sent in ml_tokenized for word in sent)

    print(f"\nEnglish vocabulary size: {len(en_vocab)}")
    print(f"Malayalam vocabulary size: {len(ml_vocab)}")

    print("\n[3] Computing P(Malayalam|English)...")
    print("-" * 80)
    en_ml_counts = compute_word_alignment_counts(en_tokenized, ml_tokenized)
    p_ml_given_en = compute_translation_probs(en_ml_counts, en_vocab)

    print(f"Total word pair co-occurrences: {len(en_ml_counts)}")
    print("\nTop translation probabilities P(ml|en):")
    sorted_probs = sorted(p_ml_given_en.items(), key=lambda x: -x[1])

    for i, ((en_word, ml_word), prob) in enumerate(sorted_probs[:20], 1):
        print(f"  {i:2d}. P({ml_word:15s} | {en_word:12s}) = {prob:.4f}")

    print("\n[4] Computing P(English|Malayalam)...")
    print("-" * 80)
    ml_en_counts = compute_word_alignment_counts(ml_tokenized, en_tokenized)
    p_en_given_ml = compute_translation_probs(ml_en_counts, ml_vocab)

    print(f"Total word pair co-occurrences: {len(ml_en_counts)}")
    print("\nTop translation probabilities P(en|ml):")
    sorted_probs = sorted(p_en_given_ml.items(), key=lambda x: -x[1])

    for i, ((ml_word, en_word), prob) in enumerate(sorted_probs[:20], 1):
        print(f"  {i:2d}. P({en_word:12s} | {ml_word:15s}) = {prob:.4f}")

    print("\n[5] Sample Translation Probability Queries:")
    print("-" * 80)

    sample_queries = [
        ("I", "ഞാൻ"),
        ("love", "സ്നേഹിക്കുന്നു"),
        ("book", "പുസ്തകം"),
        ("school", "സ്കൂളിലേക്ക്"),
        ("learning", "പഠിക്കുന്നു"),
    ]

    print("\nP(Malayalam|English):")
    for en, ml in sample_queries:
        prob = p_ml_given_en.get((en, ml), 0.0)
        print(f"  P({ml:15s} | {en:12s}) = {prob:.4f}")

    print("\nP(English|Malayalam):")
    for en, ml in sample_queries:
        prob = p_en_given_ml.get((ml, en), 0.0)
        print(f"  P({en:12s} | {ml:15s}) = {prob:.4f}")

    print("=" * 80)


if __name__ == '__main__':
    main()
