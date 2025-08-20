from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, List

# Pre-compile a regex that matches the longest / most specific token types first.
# Order matters: more specific patterns (acronyms, hyphenated words) precede simpler word patterns.
TOKEN_REGEX = re.compile(
		r"""
		(?:[A-Z]{2,})                               # Simple acronym (USA, NASA)
	| (?:[A-Z](?:\.[A-Z])+\.? )                 # Dotted acronym (U.S.A, U.N., E.U.)
	| (?:[A-Za-z]+(?:-[A-Za-z]+)+)               # Hyphenated words (ice-cream, mother-in-law)
	| (?:\d+(?:\.\d+)*)                       # Numbers / decimals
	| (?:[A-Za-z]+(?:'[A-Za-z]+)*)               # Words & words with apostrophe parts
	| \.{3}                                     # Ellipsis ...
	| --                                         # Double dash treated as one token
	| [\.!?;,:\(\)\[\]\{\}\"“”‘’'`]|…     # Punctuation marks (incl typographic quotes & ellipsis char)
	| [@#$%\^&*+=/\\<>~]                       # Special symbols
	| -                                          # Standalone hyphen (after hyphenated words rule)
		""",
		re.VERBOSE,
)

_CONTRACTION_SPLIT_RE = re.compile(
	r"^(?P<base>[A-Za-z]+)(?P<ending>'s|'re|'ve|'ll|'d|'m|n't)$",
	re.IGNORECASE,
)


def tokenize(text: str) -> List[str]:
	"""Tokenize English *text* according to the specified simple rule set.

	Steps:
	  1. Run master regex to grab candidate tokens.
	  2. Post-process tokens to split standard contractions into two tokens.
	  3. Emit any stray characters not covered (defensive, though regex aims for full coverage).
	"""
	tokens: List[str] = []
	position = 0
	length = len(text)

	for match in TOKEN_REGEX.finditer(text):
		start, end = match.span()
		# Capture any intervening non-whitespace chars (should be rare) individually.
		if start > position:
			gap_segment = text[position:start]
			for ch in gap_segment:
				if not ch.isspace():
					tokens.append(ch)
		raw = match.group(0)
		tokens.extend(_expand_contraction(raw))
		position = end

	# Trailing segment
	if position < length:
		tail = text[position:]
		for ch in tail:
			if not ch.isspace():
				tokens.append(ch)

	return tokens


def _expand_contraction(token: str) -> List[str]:
	m = _CONTRACTION_SPLIT_RE.match(token)
	if not m:
		return [token]
	base = m.group("base")
	ending = m.group("ending")
	# Directly return captured parts; pattern already isolates standard endings including n't without special casing.
	return [base, ending]


def main():
	# Load text from file
	text_path = Path(__file__).with_name("input_text.txt")
	content = text_path.read_text(encoding="utf-8")
	print(f"Loaded text from: {text_path}")
	print("\n--- Raw Text ---\n")
	print(content.strip())
	print("\n--- Tokens ---\n")
	tokens = tokenize(content)
	for i, tok in enumerate(tokens, 1):
		print(f"{i:>4}: {tok}")
	print(f"\nTotal tokens: {len(tokens)}")

if __name__ == "__main__":
	main()

