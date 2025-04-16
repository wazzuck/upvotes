"""
This script tokenizes text data and creates mappings between tokens and indices.
It handles unknown tokens by replacing them with a special <UNK> token.
"""

from datasets import load_dataset
import csv
from collections import Counter

# Load the token to index mapping from a CSV file
def load_tokeniser(index_to_token_filepath, token_to_index_filepath):
    """
    Loads token mappings from CSV files.
    
    Args:
        index_to_token_filepath: Path to index-to-token mapping file
        token_to_index_filepath: Path to token-to-index mapping file
    
    Returns:
        tuple: (token_to_index dictionary, index_to_token dictionary)
    """
    print("\n=== Loading Token Mappings ===")
    print("Step 1: Loading token-to-index mapping...")
    token_to_index = {}
    try:
        with open(index_to_token_filepath, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                token, index = row
                token_to_index[token] = int(index)
        print(f"Loaded {len(token_to_index)} token-to-index mappings")
    except FileNotFoundError:
        print(f"Error: File {index_to_token_filepath} not found")
        return None, None
    except Exception as e:
        print(f"Error loading token-to-index mapping: {str(e)}")
        return None, None

    print("\nStep 2: Loading index-to-token mapping...")
    index_to_token = {}
    try:
        with open(token_to_index_filepath, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                index, token = row
                index_to_token[index] = token
        print(f"Loaded {len(index_to_token)} index-to-token mappings")
    except FileNotFoundError:
        print(f"Error: File {token_to_index_filepath} not found")
        return None, None
    except Exception as e:
        print(f"Error loading index-to-token mapping: {str(e)}")
        return None, None

    print(f"\nTokeniser loaded successfully with {len(token_to_index)} tokens")
    return token_to_index, index_to_token

def tokenise_text(text, token_to_index):
    """
    Tokenizes text using the provided token mappings.
    Unknown tokens are replaced with <UNK>.
    
    Args:
        text: Input text to tokenize
        token_to_index: Dictionary mapping tokens to indices
    
    Returns:
        list: List of token indices
    """
    print("\n=== Tokenizing Text ===")
    print("Step 1: Splitting text into tokens...")
    tokens = text.split()
    print(f"Found {len(tokens)} raw tokens")
    
    print("\nStep 2: Converting tokens to indices...")
    tokenised_text = []
    unknown_count = 0
    for token in tokens:
        if token in token_to_index:
            tokenised_text.append(token_to_index[token])
        else:
            tokenised_text.append(token_to_index['<UNK>'])
            unknown_count += 1
    
    print(f"Tokenization complete:")
    print(f"  Total tokens: {len(tokenised_text)}")
    print(f"  Unknown tokens: {unknown_count}")
    print(f"  Unknown token percentage: {(unknown_count/len(tokenised_text)*100):.2f}%")
    
    return tokenised_text

if __name__ == '__main__':
    print("=== Starting Text Tokenization ===")
    
    # 1. Load the dataset
    print("\n=== Loading Dataset ===")
    try:
        print("Downloading and loading text8 dataset...")
        dataset = load_dataset("afmck/text8")
        text = dataset['train'][0]['text']
        print(f"Dataset loaded successfully. Text length: {len(text)} characters")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        exit(1)

    # 2. Load tokeniser
    print("\n=== Loading Token Mappings ===")
    token_to_index_filepath = 'data/processed/token_to_index.csv'
    index_to_token_filepath = 'data/processed/index_to_token.csv'
    token_to_index, index_to_token = load_tokeniser(token_to_index_filepath, index_to_token_filepath)
    
    if token_to_index is None or index_to_token is None:
        print("Error: Failed to load token mappings")
        exit(1)

    # 3. Tokenise the dataset
    tokenised_text = tokenise_text(text, token_to_index)
    
    # Analyze token distribution
    print("\n=== Analyzing Token Distribution ===")
    token_counter = Counter(tokenised_text)
    print(f"Number of unique tokens: {len(token_counter)}")
    print("\nMost common tokens:")
    for token, count in token_counter.most_common(10):
        print(f"  Token {token}: {count} occurrences")
    
    # 4. Save tokenised text to file
    print("\n=== Saving Tokenized Text ===")
    tokenised_text_filepath = 'data/processed/tokenised_text.txt'
    try:
        with open(tokenised_text_filepath, 'w') as file:
            for token in tokenised_text:
                file.write(f"{token} ")
        print(f"Tokenized text saved to {tokenised_text_filepath}")
    except Exception as e:
        print(f"Error saving tokenized text: {str(e)}")
        exit(1)
    
    print("\n=== Tokenization Complete ===")