# Tokenization and Word Normalization
# Write a python program that reads a paragraph and 
# - tokenizes the text into words
# - removes punctuation and converts all words to lowercase
# - performs stemming and lemmatization

import nltk
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter

def download_nltk_data():
    """Download necessary NLTK data packages"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        print("Download complete!")

def read_paragraph(filename):
    """Read paragraph from file"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None

def tokenize_text(text):
    """Tokenize text into words"""
    tokens = word_tokenize(text)
    return tokens

def remove_punctuation_and_lowercase(tokens):
    """Remove punctuation and convert to lowercase"""
    cleaned_tokens = []
    for token in tokens:
        cleaned_token = token.lower()
        if cleaned_token.isalpha():
            cleaned_tokens.append(cleaned_token)
    return cleaned_tokens

def perform_stemming(tokens):
    """Perform stemming using Porter Stemmer"""
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(token) for token in tokens]
    return stemmed_words

def perform_lemmatization(tokens):
    """Perform lemmatization using WordNet Lemmatizer"""
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_words

def display_results(original_text, tokens, cleaned_tokens, stemmed_words, lemmatized_words):
    """Display the results of text processing"""
    print("=" * 80)
    print("TOKENIZATION AND WORD NORMALIZATION RESULTS")
    print("=" * 80)
    
    print(f"\n1. ORIGINAL TEXT:")
    print(f"   {original_text}")
    
    print(f"\n2. TOKENIZATION:")
    print(f"   Total tokens: {len(tokens)}")
    print(f"   Tokens: {tokens}")
    
    print(f"\n3. AFTER REMOVING PUNCTUATION & LOWERCASE:")
    print(f"   Total words: {len(cleaned_tokens)}")
    print(f"   Words: {cleaned_tokens}")
    
    print(f"\n4. STEMMING (Porter Stemmer):")
    print(f"   Stemmed words: {stemmed_words}")
    
    print(f"\n5. LEMMATIZATION (WordNet Lemmatizer):")
    print(f"   Lemmatized words: {lemmatized_words}")
    
    print(f"\n6. COMPARISON - STEMMING vs LEMMATIZATION:")
    print(f"   {'Original':<15} {'Stemmed':<15} {'Lemmatized':<15}")
    print(f"   {'-'*15} {'-'*15} {'-'*15}")
    for i, word in enumerate(cleaned_tokens):
        stem = stemmed_words[i] if i < len(stemmed_words) else ""
        lemma = lemmatized_words[i] if i < len(lemmatized_words) else ""
        print(f"   {word:<15} {stem:<15} {lemma:<15}")
    
    print(f"\n7. WORD FREQUENCY:")
    word_freq = Counter(cleaned_tokens)
    print(f"   Most common words: {word_freq.most_common(10)}")

def main():
    """Main function to execute the tokenization and normalization process"""
    # Download required NLTK data
    download_nltk_data()
    
    # Read paragraph from file
    filename = "paragraph.txt"
    text = read_paragraph(filename)
    
    if text is None:
        return
    
    # Step 1: Tokenization
    print("Step 1: Tokenizing text...")
    tokens = tokenize_text(text)
    
    # Step 2: Remove punctuation and convert to lowercase
    print("Step 2: Removing punctuation and converting to lowercase...")
    cleaned_tokens = remove_punctuation_and_lowercase(tokens)
    
    # Step 3: Stemming
    print("Step 3: Performing stemming...")
    stemmed_words = perform_stemming(cleaned_tokens)
    
    # Step 4: Lemmatization
    print("Step 4: Performing lemmatization...")
    lemmatized_words = perform_lemmatization(cleaned_tokens)
    
    # Display results
    display_results(text, tokens, cleaned_tokens, stemmed_words, lemmatized_words)

if __name__ == "__main__":
    main()