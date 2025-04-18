"""
This script is responsible for training Word2Vec models (both CBOW and Skip-gram) on a text dataset. 
Word2Vec is a popular algorithm for learning word embeddings, which are dense vector representations of words that capture semantic meaning. 
This script loads a dataset, preprocesses it into pseudo-sentences, trains two types of Word2Vec models, and saves the resulting models to disk.
"""

import os
# Import the load_dataset function from the HuggingFace datasets library.
# This function allows us to easily download and load standard datasets for NLP tasks.
from datasets import load_dataset
# Import the Word2Vec class from the gensim library, which provides efficient implementations of Word2Vec.
from gensim.models import Word2Vec
# Import the multiprocessing module to utilize all available CPU cores for faster training.
import multiprocessing
# Import the time module to measure how long training takes.
import time
# Import tqdm for progress bars
import tqdm
# Import the base callback class from Gensim
from gensim.models.callbacks import CallbackAny2Vec

# Define a callback class to integrate with tqdm for progress reporting
class TqdmCallback(CallbackAny2Vec):
    """Callback to track progress during Gensim Word2Vec training using tqdm."""
    def __init__(self, total_epochs, description="Training Progress"):
        self.epoch = 0
        self.total_epochs = total_epochs
        # Initialize tqdm progress bar
        self.pbar = tqdm.tqdm(total=self.total_epochs, desc=description, unit="epoch")

    def on_epoch_end(self, model):
        # Increment epoch counter and update progress bar
        self.epoch += 1
        self.pbar.update(1)
        # You could optionally add loss reporting here if compute_loss=True in Word2Vec
        # loss = model.get_latest_training_loss()
        # self.pbar.set_postfix(loss=f'{loss:.4f}')

    def on_train_end(self, model):
        # Ensure the progress bar is closed when training finishes
        self.pbar.close()

# Define the directory to save models
MODEL_DIR = "data/models"
# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. Load the dataset ---
print("--- Starting Word2Vec Training ---")
print("[Phase 1/4] Loading text8 dataset from HuggingFace...")
# Here we use the 'afmck/text8' dataset from HuggingFace Datasets.
# This is a preprocessed, cleaned version of Wikipedia text, often used for word embedding experiments.
start_time = time.time()
dataset = load_dataset("afmck/text8", trust_remote_code=True) # Added trust_remote_code=True for newer datasets versions
print(f"Dataset loaded successfully in {time.time() - start_time:.2f} seconds.")
print(f"Dataset structure: {dataset}")

# --- 2. Preprocess: split into words, chunk into pseudo-sentences ---
print("\n[Phase 2/4] Preprocessing dataset...")
start_time = time.time()
# The dataset is loaded as a dictionary with splits (e.g., 'train').
# We extract the text from the first (and only) entry in the 'train' split.
text = dataset['train'][0]['text']
print(f"Dataset contains {len(text):,} characters.")
# Split the text into individual word tokens using whitespace as the delimiter.
tokens = text.split()
print(f"Split text into {len(tokens):,} tokens.")
print(f"First 10 tokens: {tokens[:10]}")

# Word2Vec expects a list of sentences, where each sentence is a list of words.
# The text8 dataset is just one long string, so we break it into chunks of 1000 words each to simulate sentences.
chunk_size = 1000  # Number of words per pseudo-sentence
sentences = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
print(f"Created {len(sentences)} pseudo-sentences of chunk size {chunk_size}.")
if sentences:
    print(f"First pseudo-sentence (first 20 tokens): {sentences[0][:20]}")

print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds.")

# --- 3. Train the Word2Vec models ---
N_EPOCHS = 5 # Define number of epochs

# --- CBOW Model Training ---
print("\n[Phase 3/4] Training CBOW model...")
# Record the start time to measure training duration.
start_time = time.time()
# Initialize the progress bar callback
cbow_callback = TqdmCallback(N_EPOCHS, description="CBOW Training")

# Initialize and train the Word2Vec model using the CBOW (Continuous Bag of Words) architecture.
# Parameters:
# - sentences: the list of pseudo-sentences (each a list of words)
# - vector_size: the dimensionality of the word vectors (100 is typical)
# - window: the maximum distance between the current and predicted word within a sentence (context window)
# - min_count: ignore all words with total frequency lower than this (prunes rare words)
# - workers: number of CPU cores to use for parallelization
# - sg: 0 means CBOW, 1 means Skip-gram
# - epochs: number of training iterations over the corpus
# - callbacks: list of callbacks to use during training
cbow_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=multiprocessing.cpu_count(),
    sg=0,  # Use CBOW architecture
    epochs=N_EPOCHS,
    callbacks=[cbow_callback] # Pass the tqdm callback
)
# Calculate how long training took.
cbow_elapsed = time.time() - start_time
# The progress bar replaces the need for a separate completion message here
# print(f"CBOW training completed in {cbow_elapsed:.2f} seconds.")
print(f"CBOW training wall time: {cbow_elapsed:.2f} seconds.") # Print wall time separately
print(f"CBOW model vocabulary size: {len(cbow_model.wv.index_to_key):,}")

# Save CBOW model
print("Saving CBOW model...")
cbow_save_path = os.path.join(MODEL_DIR, "word2vec_text8_cbow.model")
cbow_model.save(cbow_save_path)
print(f"CBOW model saved to '{cbow_save_path}'.")

# --- Skip-gram Model Training ---
print("\n[Phase 4/4] Training Skip-gram model...")
# Record the start time to measure training duration.
start_time = time.time()
# Initialize the progress bar callback
skipgram_callback = TqdmCallback(N_EPOCHS, description="Skip-gram Training")

# Initialize and train the Word2Vec model using the Skip-gram architecture (sg=1).
# Skip-gram tries to predict the context words given a target word. Often better for rare words but slower.
skipgram_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=multiprocessing.cpu_count(),
    sg=1,  # Use Skip-gram architecture
    epochs=N_EPOCHS,
    callbacks=[skipgram_callback] # Pass the tqdm callback
)
# Calculate how long training took.
skipgram_elapsed = time.time() - start_time
# The progress bar replaces the need for a separate completion message here
# print(f"Skip-gram training completed in {skipgram_elapsed:.2f} seconds.")
print(f"Skip-gram training wall time: {skipgram_elapsed:.2f} seconds.") # Print wall time separately
print(f"Skip-gram model vocabulary size: {len(skipgram_model.wv.index_to_key):,}")

# Save Skip-gram model
print("Saving Skip-gram model...")
skipgram_save_path = os.path.join(MODEL_DIR, "word2vec_text8_skipgram.model")
skipgram_model.save(skipgram_save_path)
print(f"Skip-gram model saved to '{skipgram_save_path}'.")

print("\n--- Word2Vec Training Finished ---")
