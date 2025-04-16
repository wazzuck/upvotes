"""
This script is responsible for tokenizing a text dataset, counting word frequencies, filtering out rare words, and saving the results to files. 
It is a foundational step in many NLP (Natural Language Processing) pipelines, as it prepares the data for further processing or model training.
"""

# Import the regular expressions module, which allows us to use regex for advanced string processing.
import re
# Import the load_dataset function from the HuggingFace datasets library.
# This function allows us to easily download and load standard datasets for NLP tasks.
from datasets import load_dataset

# Define a function to tokenize text and count word frequencies.
def tokenize_text(text):
    """
    Tokenizes the input text into a list of words and their frequency.

    Args:
        text (str): The input text to tokenize.

    Returns:
        dict: A dictionary of lowercase words (with punctuation removed), mapped to their frequency.
    """

    # Use a regular expression to find all word tokens in the text.
    # \b\w+\b matches sequences of word characters (letters, digits, underscores) between word boundaries.
    # text.lower() converts all characters to lowercase to ensure case-insensitive matching.
    tokens = re.findall(r'\b\w+\b', text.lower())
    
    # Create a dictionary to store the frequency of each token.
    word_freq = {}
    for token in tokens:
        # If the token is already in the dictionary, increment its count.
        if token in word_freq:
            word_freq[token] += 1
        # If the token is not in the dictionary, add it with a count of 1.
        else:
            word_freq[token] = 1
    
    # Return the dictionary mapping tokens to their frequencies.
    return word_freq

# 1. Load the dataset
print("\n=== Loading Dataset ===")
# Here we use the 'afmck/text8' dataset from HuggingFace Datasets.
# This is a preprocessed, cleaned version of Wikipedia text, often used for word embedding experiments.
dataset = load_dataset("afmck/text8")
print(f"Dataset loaded. Size: {len(dataset['train'])} examples")

# 2. Preprocess the dataset
print("\n=== Preprocessing Dataset ===")
# The dataset is loaded as a dictionary with splits (e.g., 'train').
# We extract the text from the first (and only) entry in the 'train' split.
text = dataset['train'][0]['text']
print(f"First 200 characters of text: {text[:200]}...")
print(f"Total text length: {len(text)} characters")

# 3. Tokenize the dataset
print("\n=== Tokenization ===")
# We call our tokenize_text function to get a dictionary of word frequencies from the text.
tokens = tokenize_text(text)
print(f"Tokenized text sample (first 10 items): {dict(list(tokens.items())[:10])}")

# 4. Print some data on the tokens
print("\n=== Token Statistics ===")
print(f"Total number of words (including duplicates): {sum(tokens.values())}")
print(f"Most common token: {max(tokens, key=tokens.get)}")
print(f"Frequency of most common token: {tokens[max(tokens, key=tokens.get)]}")
print(f"Average frequency per token: {sum(tokens.values()) / len(tokens):.2f}")

# 5. Set a threshold below which tokens are removed from the tokens
print("\n=== Filtering Tokens ===")
threshold = 5  # Only keep words that appear at least 5 times in the dataset
print(f"Filtering out tokens that appear less than {threshold} times...")

# Filter out tokens that appear less than the threshold
filtered_tokens = {word: freq for word, freq in tokens.items() if freq >= threshold}
print(f"Number of tokens before filtering: {len(tokens)}")
print(f"Number of tokens after filtering: {len(filtered_tokens)}")
print(f"Percentage of tokens kept: {(len(filtered_tokens) / len(tokens) * 100):.2f}%")
print(f"Sample of filtered tokens (first 10): {dict(list(filtered_tokens.items())[:10])}")

# 6. Save filtered tokens to a data file
print("\n=== Saving Results ===")
output_file_path = "data/processed/filtered_tokens.txt"
print(f"Saving filtered tokens to {output_file_path}...")
# Write filtered tokens to a text file, one word and its frequency per line
with open(output_file_path, 'w') as output_file:
    for word, freq in filtered_tokens.items():
        output_file.write(f"{word} {freq}\n")
print(f"Filtered tokens saved to {output_file_path}")

# 7. Save the filtered tokens to a CSV file
import pandas as pd
output_csv_path = "data/processed/filtered_tokens.csv"
print(f"Saving filtered tokens to {output_csv_path}...")
# Convert filtered tokens to a DataFrame and save as CSV
filtered_tokens_df = pd.DataFrame(list(filtered_tokens.items()), columns=['word', 'frequency'])
filtered_tokens_df.to_csv(output_csv_path, index=False)
print(f"Filtered tokens saved to {output_csv_path}")
print("\n=== Processing Complete ===")
