# Translation Probability Computation

Computes bidirectional translation probabilities P(f|e) and P(e|f) from an English-Malayalam parallel corpus.

## Corpus

- 5 manually constructed sentence pairs
- English ↔ Malayalam translations

## Method

- Tokenization by whitespace
- Co-occurrence counting for all word pairs in aligned sentences
- Probability: P(target|source) = count(source,target) / Σ count(source,\*)

## Run

```pwsh
python main.py
```

## Output

- Parallel corpus display
- Top translation probabilities P(Malayalam|English)
- Top translation probabilities P(English|Malayalam)
- Sample queries for specific word pairs
- Analysis and limitations

No dependencies required (uses only Python standard library).
