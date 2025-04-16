"""
This script is responsible for training Word2Vec models (both CBOW and Skip-gram) on a text dataset. 
Word2Vec is a popular algorithm for learning word embeddings, which are dense vector representations of words that capture semantic meaning. 
This script loads a dataset, preprocesses it into pseudo-sentences, trains two types of Word2Vec models, and saves the resulting models to disk.
"""

# Import the load_dataset function from the HuggingFace datasets library.
# This function allows us to easily download and load standard datasets for NLP tasks.
from datasets import load_dataset
# Import the Word2Vec class from the gensim library, which provides efficient implementations of Word2Vec.
from gensim.models import Word2Vec
# Import the multiprocessing module to utilize all available CPU cores for faster training.
import multiprocessing
# Import the time module to measure how long training takes.
import time

# 1. Load the dataset
print("Loading dataset...")
# Here we use the 'afmck/text8' dataset from HuggingFace Datasets.
# This is a preprocessed, cleaned version of Wikipedia text, often used for word embedding experiments.
dataset = load_dataset("afmck/text8")

# 2. Preprocess: split into words, chunk into pseudo-sentences
print("Preprocessing dataset...")
# The dataset is loaded as a dictionary with splits (e.g., 'train').
# We extract the text from the first (and only) entry in the 'train' split.
text = dataset['train'][0]['text']
# Split the text into individual word tokens using whitespace as the delimiter.
tokens = text.split()
# Word2Vec expects a list of sentences, where each sentence is a list of words.
# The text8 dataset is just one long string, so we break it into chunks of 1000 words each to simulate sentences.
chunk_size = 1000  # Number of words per pseudo-sentence
sentences = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
print(f"Created {len(sentences)} pseudo-sentences.")

# 3. Train the Word2Vec models

# Train CBOW model
print("Training CBOW model...")
# Record the start time to measure training duration.
start_time = time.time()
# Initialize and train the Word2Vec model using the CBOW (Continuous Bag of Words) architecture.
# Parameters:
# - sentences: the list of pseudo-sentences (each a list of words)
# - vector_size: the dimensionality of the word vectors (100 is typical)
# - window: the maximum distance between the current and predicted word within a sentence
# - min_count: ignore all words with total frequency lower than this
# - workers: number of CPU cores to use
# - sg: 0 means CBOW, 1 means Skip-gram
cbow_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=multiprocessing.cpu_count(),
    sg=0  # CBOW
)
# Calculate how long training took.
cbow_elapsed = time.time() - start_time
print(f"CBOW training completed in {cbow_elapsed:.2f} seconds.")

# Save CBOW model
# Save the trained CBOW model to disk for later use.
cbow_model.save("models/word2vec_text8_cbow.model")
print("CBOW model saved to 'word2vec_text8_cbow.model'.")

# Train Skip-gram model
print("Training Skip-gram model...")
# Record the start time to measure training duration.
start_time = time.time()
# Initialize and train the Word2Vec model using the Skip-gram architecture (sg=1).
skipgram_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=multiprocessing.cpu_count(),
    sg=1  # Skip-gram
)
# Calculate how long training took.
skipgram_elapsed = time.time() - start_time
print(f"Skip-gram training completed in {skipgram_elapsed:.2f} seconds.")

# Save Skip-gram model
# Save the trained Skip-gram model to disk for later use.
skipgram_model.save("models/word2vec_text8_skipgram.model")
print("Skip-gram model saved to 'word2vec_text8_skipgram.model'.")
