"""
Question 2: Extract digits and phone numbers from a string using regular expressions

This program demonstrates:
1. Extracting all digits from a string
2. Extracting 10-digit phone numbers from a string

Input: Text from a file
Output: Lists of extracted digits and phone numbers
"""

import re

def extract_digits(text):
    """
    Extract all individual digits from the text

    Args:
        text (str): Input text

    Returns:
        list: List of all individual digits found
    """
    # Pattern to match individual digits
    digit_pattern = r'\d'
    digits = re.findall(digit_pattern, text)
    return digits

def extract_all_numbers(text):
    """
    Extract all sequences of digits from the text

    Args:
        text (str): Input text

    Returns:
        list: List of all number sequences found
    """
    # Pattern to match sequences of digits
    number_pattern = r'\d+'
    numbers = re.findall(number_pattern, text)
    return numbers

def extract_phone_numbers(text):
    """
    Extract 10-digit phone numbers from the text

    Args:
        text (str): Input text

    Returns:
        list: List of 10-digit phone numbers found
    """
    # Pattern to match exactly 10 consecutive digits
    phone_pattern = r'\b\d{10}\b'
    phone_numbers = re.findall(phone_pattern, text)
    return phone_numbers

def extract_formatted_phone_numbers(text):
    """
    Extract phone numbers in various formats (10 digits)

    Args:
        text (str): Input text

    Returns:
        list: List of phone numbers in different formats
    """
    # Pattern to match phone numbers with various formatting
    # Matches: (555) 123-4567, 555-123-4567, 555.123.4567, 5551234567
    formatted_phone_pattern = r'(?:\(\d{3}\)\s?|\d{3}[-.\s]?)?\d{3}[-.\s]?\d{4}'
    formatted_phones = re.findall(formatted_phone_pattern, text)
    return formatted_phones

def main():
    # Read input from file
    try:
        with open('input_text.txt', 'r', encoding='utf-8') as file:
            text = file.read()

        print("Original Text:")
        print("-" * 50)
        print(text)
        print("\n" + "=" * 50)

        # Part 1: Extract digits
        print("\n1. DIGIT EXTRACTION:")
        print("-" * 30)

        # Extract individual digits
        individual_digits = extract_digits(text)
        print(f"Individual digits found: {individual_digits}")
        print(f"Total individual digits: {len(individual_digits)}")

        # Extract number sequences
        number_sequences = extract_all_numbers(text)
        print(f"Number sequences found: {number_sequences}")
        print(f"Total number sequences: {len(number_sequences)}")

        # Part 2: Extract phone numbers
        print("\n2. PHONE NUMBER EXTRACTION:")
        print("-" * 35)

        # Extract 10-digit phone numbers
        phone_numbers = extract_phone_numbers(text)
        print(f"10-digit phone numbers found: {phone_numbers}")
        print(f"Total 10-digit phone numbers: {len(phone_numbers)}")

        # Extract formatted phone numbers
        formatted_phones = extract_formatted_phone_numbers(text)
        print(f"Formatted phone numbers found: {formatted_phones}")
        print(f"Total formatted phone numbers: {len(formatted_phones)}")

        # Additional analysis
        print("\n3. DETAILED ANALYSIS:")
        print("-" * 25)

        # Count digits by frequency
        digit_count = {}
        for digit in individual_digits:
            digit_count[digit] = digit_count.get(digit, 0) + 1

        print("Digit frequency:")
        for digit, count in sorted(digit_count.items()):
            print(f"  Digit '{digit}': {count} times")

        # Validate phone numbers
        print("\nPhone number validation:")
        for phone in phone_numbers:
            if len(phone) == 10:
                print(f"  {phone}: Valid 10-digit phone number")
            else:
                print(f"  {phone}: Invalid length ({len(phone)} digits)")

    except FileNotFoundError:
        print("Error: input_text.txt file not found!")
        print("Please create an input file with text containing digits and phone numbers.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
