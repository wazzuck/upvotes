"""
This script is responsible for creating a tokeniser from a text dataset. 
A tokeniser is a mapping from words (tokens) to unique integer indices, 
which is a common preprocessing step in NLP (Natural Language Processing) tasks. 
This script loads a dataset, tokenizes the text, filters out rare words,
assigns indices to the remaining tokens, and saves the mappings to files for later use in model training.
"""

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

    # Split the text into tokens (words) using whitespace as the delimiter.
    # Note: This is a very simple tokenizer and does not handle punctuation or special cases.
    tokens = text.split()
    print(f"[Tokenization] First 20 tokens: {tokens[:20]}")  # Show a sample of the tokens

    # Create a dictionary to store the frequency of each token.
    word_freq = {}
    for token in tokens:
        # If the token is already in the dictionary, increment its count.
        if token in word_freq:
            word_freq[token] += 1
        # If the token is not in the dictionary, add it with a count of 1.
        else:
            word_freq[token] = 1
    print(f"[Tokenization] First 10 word frequencies: {list(word_freq.items())[:10]}")  # Show a sample of word frequencies
    print(f"[Tokenization] Total unique tokens: {len(word_freq)}")
    # Return the dictionary mapping tokens to their frequencies.
    return word_freq

# 1. Load the dataset
print("[Stage 1] Loading dataset...")
# Here we use the 'afmck/text8' dataset from HuggingFace Datasets.
# This is a preprocessed, cleaned version of Wikipedia text, often used for word embedding experiments.
dataset = load_dataset("afmck/text8")

print(f"[Stage 1] Dataset keys: {list(dataset.keys())}")
print(f"[Stage 1] Example entry: {str(dataset['train'][0])[:200]}...")  # Show a sample of the data

# 2. Preprocess the dataset
print("[Stage 2] Preprocessing dataset...")
# The dataset is loaded as a dictionary with splits (e.g., 'train').
# We extract the text from the first (and only) entry in the 'train' split.
text = dataset['train'][0]['text']
print(f"[Stage 2] First 200 characters of text: {text[:200]}...")

# 3. Tokenize the dataset
print("[Stage 3] Tokenising dataset...")
# We call our tokenize_text function to get a dictionary of word frequencies from the text.
tokens = tokenize_text(text)

# 4. Print some data on the tokens
# These print statements are for exploratory purposes, to understand the vocabulary and frequency distribution.
print(f"[Stage 4] Number of tokens: {len(tokens)}")  # Total number of unique tokens (words)
print(f"[Stage 4] First 10 tokens: {list(tokens.items())[:10]}")  # Show the first 10 tokens and their frequencies
print(f"[Stage 4] Number of unique tokens: {len(set(tokens))}")  # Should be the same as above
print(f"[Stage 4] Most common token: {max(tokens, key=tokens.get)}")  # The word with the highest frequency
print(f"[Stage 4] Frequency of most common token: {tokens[max(tokens, key=tokens.get)]}")  # Its frequency

# 5. Set a threshold below which tokens are removed from the tokens
# Many words in natural language are rare. To reduce noise and model size, we filter out words that appear less than 'threshold' times.
threshold = 70  # Only keep words that appear at least 70 times in the dataset
print(f"[Stage 5] Filtering tokens with frequency < {threshold}...")

# Create a new dictionary containing only the tokens that meet the frequency threshold.
filtered_tokens = {word: freq for word, freq in tokens.items() if freq >= threshold}
print(f"[Stage 5] Number of tokens after filtering: {len(filtered_tokens)}")  # Number of tokens left after filtering
print(f"[Stage 5] First 10 filtered tokens: {list(filtered_tokens.items())[:10]}")  # Show the first 10 filtered tokens
print(f"[Stage 5] Number of unique filtered tokens: {len(set(filtered_tokens))}")  # Should be the same as above

# 6. Assign an index to each token
# Machine learning models require numerical input, so we map each token to a unique integer index.
# This is called a 'token-to-index' mapping.
print("[Stage 6] Creating token-to-index and index-to-token mappings...")
token_to_index = {token: index for index, token in enumerate(filtered_tokens.keys())}
print(f"[Stage 6] First 10 token-to-index mappings: {list(token_to_index.items())[:10]}")  # Show the first 10 mappings
# We also create the reverse mapping: index to token.
index_to_token = {index: token for token, index in token_to_index.items()}
print(f"[Stage 6] First 10 index-to-token mappings: {list(index_to_token.items())[:10]}")  # Show the first 10 reverse mappings

# 7. Add to the tokeniser <UNK> for unknown tokens
# In practice, you will encounter words in new data that were not in your training set.
# We add a special token '<UNK>' (unknown) to handle these cases.
token_to_index['<UNK>'] = len(token_to_index)  # Assign the next available index to <UNK>
index_to_token[len(index_to_token)] = '<UNK>'  # Add the reverse mapping
print(f"[Stage 7] Added <UNK> token. Total tokens now: {len(token_to_index)}")

# 8. Save filtered tokens to a data file
# We save the token-to-index mapping to a CSV file for later use in model training or inference.
output_file_path = "data/processed/token_to_index.csv"
with open(output_file_path, 'w') as file:
    for token, index in token_to_index.items():
        file.write(f"{token},{index}\n")
print(f"[Stage 8] Filtered tokens saved to {output_file_path}.")

# 8. Save filtered indexes to a data file
# We also save the index-to-token mapping to a separate CSV file.
output_file_path = "data/processed/index_to_token.csv"
with open(output_file_path, 'w') as file:
    for index, token in index_to_token.items():
        file.write(f"{index},{token}\n")
print(f"[Stage 8] Index to token mapping saved to {output_file_path}.")