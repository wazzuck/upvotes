import re
from datasets import load_dataset

# Function to tokenize text and count word frequencies
def tokenize_text(text):
    """
    Tokenizes the input text into a list of words and their frequency.

    Args:
        text (str): The input text to tokenize.

    Returns:
        list: A dictionary of lowercase words with punctuation removed, mapped
        to their frequency
    """

    # Remove punctuation, split into words, and convert to lowercase
    tokens = re.findall(r'\b\w+\b', text.lower())
    
    # Create a dictionary with word frequencies
    word_freq = {}
    for token in tokens:
        if token in word_freq:
            word_freq[token] += 1
        else:
            word_freq[token] = 1
    
    return word_freq

# 1. Load the dataset
print("Loading dataset...")
# Download and load the 'text8' dataset from HuggingFace Datasets
dataset = load_dataset("afmck/text8")

# 2. Preprocess the dataset
print("Preprocessing dataset...")
# Extract the text from the training split of the dataset
text = dataset['train'][0]['text']

# 3. Tokenize the dataset
print("Starting tokenization...")
# Tokenize the text and get a dictionary of word frequencies
tokens = tokenize_text(text)
print(f"Tokenized text: {tokens}")

# 4. Print some data on the tokens
print(f"Number of tokens: {len(tokens)}")  # Number of unique tokens
print(f"First 10 tokens: {list(tokens.items())[:10]}")  # Show first 10 tokens and their frequencies
print(f"Number of unique tokens: {len(set(tokens))}")  # Redundant, same as above
print(f"Most common token: {max(tokens, key=tokens.get)}")  # Token with highest frequency
print(f"Frequency of most common token: {tokens[max(tokens, key=tokens.get)]}")  # Its frequency

# 5. Set a treshold below which tokens are removed from the tokens
threshold = 5  # Minimum frequency for a token to be kept

# Filter out tokens that appear less than the threshold
filtered_tokens = {word: freq for word, freq in tokens.items() if freq >= threshold}
print(f"Number of tokens after filtering: {len(filtered_tokens)}")
print(f"First 10 filtered tokens: {list(filtered_tokens.items())[:10]}")
print(f"Number of unique filtered tokens: {len(set(filtered_tokens))}")

# 6. Save filtered tokens to a data file
output_file_path = "data/processed/filtered_tokens.txt"
# Write filtered tokens to a text file, one word and its frequency per line
with open(output_file_path, 'w') as output_file:
    for word, freq in filtered_tokens.items():
        output_file.write(f"{word} {freq}\n")
print(f"Filtered tokens saved to {output_file_path}.")

# 7. Save the filtered tokens to a CSV file
import pandas as pd
output_csv_path = "data/processed/filtered_tokens.csv"
# Convert filtered tokens to a DataFrame and save as CSV
filtered_tokens_df = pd.DataFrame(list(filtered_tokens.items()), columns=['word', 'frequency'])
filtered_tokens_df.to_csv(output_csv_path, index=False)
print(f"Filtered tokens saved to {output_csv_path}.")
