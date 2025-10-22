from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import stanza
import pandas as pd


def ensure_models():
    stanza.download('hi', verbose=False)
    stanza.download('en', verbose=False)


def build_hi_df(nlp, sentences):
    rows = []
    for s in sentences:
        doc = nlp(s)
        for sent in doc.sentences:
            for w in sent.words:
                rows.append({
                    'sent': s,
                    'word': w.text,
                    'lemma': w.lemma,
                    'pos': w.pos,
                    'xpos': w.xpos,
                    'feats': w.feats or 'None',
                })
    return pd.DataFrame(rows)


def main():
    out_dir = Path(__file__).parent

    print("=" * 80)
    print("HINDI POS TAGGING WITH STANZA")
    print("=" * 80)

    print("\n[1/5] Loading models...")
    ensure_models()
    nlp_hi = stanza.Pipeline('hi', processors='tokenize,pos,lemma', verbose=False)
    nlp_en = stanza.Pipeline('en', processors='tokenize,pos,lemma', verbose=False)
    print("✓ Models loaded")

    hindi_sentences = [
        "राम स्कूल जाता है।",
        "सीता ने किताब पढ़ी।",
        "वह बहुत सुंदर है।",
        "मैं आज बाजार जाऊंगा।",
        "बच्चे पार्क में खेल रहे हैं।",
        "यह एक बड़ा घर है।",
        "मेरा नाम राज है।",
        "वे लोग दिल्ली से आए हैं।",
        "गीता गाना गा रही है।",
        "मुझे आम बहुत पसंद है।",
    ]

    print("\n[2/5] Hindi sentences:")
    for i, s in enumerate(hindi_sentences, 1):
        print(f"  {i:2d}. {s}")

    print("\n[3/5] Tagging Hindi text...")
    hi_df = build_hi_df(nlp_hi, hindi_sentences)
    print(f"✓ Tagged {len(hi_df)} tokens\n")

    print("Sample tagged output (first 15 tokens):")
    print("-" * 80)
    print(hi_df.head(15).to_string(index=False))
    print("-" * 80)

    hi_counts = Counter(hi_df['pos'])
    print("\n[4/5] Hindi POS distribution:")
    for pos, count in sorted(hi_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (count * 2)
        print(f"  {pos:<8} {count:3d} {bar}")

    print("\n[5/5] Comparing with English...")
    en_sentences = [
        "Ram goes to school.",
        "Sita read a book.",
        "She is very beautiful.",
        "I will go to the market today.",
        "Children are playing in the park.",
        "This is a big house.",
        "My name is Raj.",
        "Those people have come from Delhi.",
        "Geeta is singing a song.",
        "I like mangoes very much.",
    ]

    rows = []
    for s in en_sentences:
        doc = nlp_en(s)
        for sent in doc.sentences:
            for w in sent.words:
                rows.append({'sent': s, 'word': w.text, 'pos': w.pos})
    en_df = pd.DataFrame(rows)
    en_counts = Counter(en_df['pos'])

    print("\nEnglish POS distribution:")
    for pos, count in sorted(en_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (count * 2)
        print(f"  {pos:<8} {count:3d} {bar}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Hindi:   {len(hi_df)} tokens, {len(hi_counts)} unique POS tags")
    print(f"English: {len(en_df)} tokens, {len(en_counts)} unique POS tags")

if __name__ == '__main__':
    main()
