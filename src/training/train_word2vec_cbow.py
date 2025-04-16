"""
This script implements a Continuous Bag of Words (CBOW) Word2Vec model for learning word embeddings.
The CBOW model predicts a target word based on its surrounding context words.

Key components:
1. Data loading and preprocessing
2. CBOW model architecture
3. Training loop
4. Model saving

The script processes text data to create word embeddings that capture semantic
relationships between words.
"""

import csv
from datasets import load_dataset
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
import multiprocessing
import os

# Set up multiprocessing
def get_num_workers():
    """Get the optimal number of workers for data loading"""
    num_workers = multiprocessing.cpu_count()
    # Leave some CPU for the main process and other tasks
    num_workers = min(num_workers - 1, 8)  # Cap at 8 workers
    return max(1, num_workers)  # Ensure at least 1 worker

# Set PyTorch to use all available threads
torch.set_num_threads(multiprocessing.cpu_count())

def load_tokeniser(index_to_token_filepath, token_to_index_filepath):
    """
    Loads the token-to-index and index-to-token mappings from CSV files.
    These mappings are essential for converting between words and their numerical representations.
    
    Args:
        index_to_token_filepath: Path to the file containing index-to-token mappings
        token_to_index_filepath: Path to the file containing token-to-index mappings
    
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

def load_tokenised_text(filepath):
    """
    Loads the pre-tokenized text from a file.
    
    Args:
        filepath: Path to the file containing tokenized text
    
    Returns:
        list: List of token indices representing the text
    """
    print("\n=== Loading Tokenized Text ===")
    print(f"Loading tokenized text from {filepath}...")
    with open(filepath, 'r') as file:
        tokenised_text = file.read().split()
    print(f"Loaded {len(tokenised_text)} tokens")
    print("\nText Statistics:")
    print(f"Total tokens: {len(tokenised_text)}")
    print(f"Unique tokens: {len(set(tokenised_text))}")
    print("\nSample of first 20 tokens with their indices:")
    for i, token in enumerate(tokenised_text[:20]):
        print(f"  Position {i}: Token {token}")
    return tokenised_text

class Word2VecCBOW(nn.Module):
    """
    Continuous Bag of Words (CBOW) model for Word2Vec.
    The model learns word embeddings by predicting a target word from its context.
    """
    def __init__(self, vocab_size, embedding_dim):
        """
        Initialize the CBOW model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the word embeddings
        """
        super(Word2VecCBOW, self).__init__()
        print(f"\nInitializing CBOW model with:")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Embedding dimension: {embedding_dim}")
        
        # Embedding layer converts word indices to dense vectors
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Linear layer for predicting the target word
        self.linear = nn.Linear(embedding_dim, vocab_size)
        print("Model architecture initialized successfully")

    def forward(self, context):
        """
        Forward pass of the model.
        
        Args:
            context: Tensor of context word indices
        
        Returns:
            Tensor: Logits for predicting the target word
        """
        # Average the embeddings of context words
        embedded = self.embeddings(context).mean(dim=1)
        # Predict the target word
        out = self.linear(embedded)
        return out

class CBOWDataset(Dataset):
    """
    Custom dataset for CBOW training.
    Creates context-target pairs from the tokenized text.
    """
    def __init__(self, tokenised_text, context_size, token_to_index):
        """
        Initialize the dataset.
        
        Args:
            tokenised_text: List of token indices
            context_size: Number of words to consider on each side of the target word
            token_to_index: Dictionary mapping tokens to indices
        """
        self.tokenised_text = tokenised_text
        self.context_size = context_size
        self.token_to_index = token_to_index
        print(f"\n=== Creating CBOW Dataset ===")
        print(f"Context size: {context_size}")
        print("Creating context-target pairs...")
        self.data = self.create_cbow_data()
        print(f"Created {len(self.data)} context-target pairs")
        print("\nSample of first 5 context-target pairs:")
        for i, (context, target) in enumerate(self.data[:5]):
            print(f"Pair {i+1}:")
            print(f"  Context: {context}")
            print(f"  Target: {target}")

    def create_cbow_data(self):
        """
        Creates context-target pairs for CBOW training.
        
        Returns:
            list: List of (context, target) pairs
        """
        data = []
        for i in range(self.context_size, len(self.tokenised_text) - self.context_size):
            # Get context words (before and after target)
            context = self.tokenised_text[i - self.context_size:i] + self.tokenised_text[i + 1:i + self.context_size + 1]
            target = self.tokenised_text[i]
            if len(context) == 2 * self.context_size and target is not None:
                # Convert tokens to indices
                context_indices = [self.token_to_index.get(token, self.token_to_index['<UNK>']) for token in context]
                target_index = self.token_to_index.get(target, self.token_to_index['<UNK>'])
                data.append((context_indices, target_index))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        # Convert to tensors - they're already integers at this point
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

if __name__ == '__main__':
    print("=== Starting Word2Vec CBOW Training ===")
    print(f"Number of CPU cores: {multiprocessing.cpu_count()}")
    print(f"Number of PyTorch threads: {torch.get_num_threads()}")
    
    # 1. Load tokens
    print("\n=== Loading Token Mappings ===")
    token_to_index_filepath = 'data/processed/token_to_index.csv'
    index_to_token_filepath = 'data/processed/index_to_token.csv'
    token_to_index, index_to_token = load_tokeniser(token_to_index_filepath, index_to_token_filepath)
    
    # 2. Load tokenised text
    print("\n=== Loading Tokenized Text ===")
    tokenised_text_filepath = 'data/processed/tokenised_text.txt'
    tokenised_text = load_tokenised_text(tokenised_text_filepath)

    # 3. Prepare CBOW dataset
    print("\n=== Preparing CBOW Dataset ===")
    context_size = 2
    print(f"Creating dataset with context size {context_size}...")
    cbow_dataset = CBOWDataset(tokenised_text, context_size, token_to_index)
    print(f"Dataset size: {len(cbow_dataset)} samples")
    
    # Create data loader with optimized settings
    print("\nCreating data loader...")
    batch_size = 256
    num_workers = get_num_workers()
    print(f"Using {num_workers} workers for data loading")
    
    dataloader = DataLoader(
        cbow_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Faster data transfer to GPU
        prefetch_factor=2,  # Prefetch 2 batches per worker
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Test a sample batch
    print("\nTesting sample batch...")
    try:
        sample_batch = next(iter(dataloader))
        context_batch, target_batch = sample_batch
        print(f"Context batch shape: {context_batch.shape}")
        print(f"Target batch shape: {target_batch.shape}")
        print(f"Sample context: {context_batch[0]}")
        print(f"Sample target: {target_batch[0]}")
    except Exception as e:
        print(f"Error testing sample batch: {str(e)}")
        exit(1)

    # 4. Initialize model and training components
    print("\n=== Initializing Model ===")
    vocab_size = len(token_to_index)
    embedding_dim = 100
    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {embedding_dim}")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Initialize model
    model = Word2VecCBOW(vocab_size, embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Model initialized successfully")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 5. Train the model
    print("\n=== Starting Training ===")
    num_epochs = 5
    print(f"Number of epochs: {num_epochs}")
    
    # Enable cuDNN benchmarking for faster training
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        start_epoch_time = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        with tqdm(total=len(dataloader), desc=f"Training") as pbar:
            for batch_idx, (context, target) in enumerate(dataloader):
                start_batch_time = time.time()
                
                # Move data to device
                context, target = context.to(device, non_blocking=True), target.to(device, non_blocking=True)
                
                # Forward pass
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                output = model(context)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update statistics
                epoch_loss += loss.item()
                batch_time = time.time() - start_batch_time
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'batch_time': f"{batch_time:.4f}s",
                    'avg_loss': f"{epoch_loss/(batch_idx+1):.4f}"
                })
        
        # Print epoch summary
        epoch_time = time.time() - start_epoch_time
        avg_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Time taken: {epoch_time:.2f} seconds")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    # 6. Save the trained model
    print("\n=== Saving Model ===")
    model_path = 'data/models/word2vec_cbow.pth'
    print(f"Saving model to {model_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'context_size': context_size
    }, model_path)
    print("Model saved successfully")
    
    print("\n=== Training Complete ===")
    print("Final model statistics:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Context size: {context_size}")
    print(f"  Total training time: {time.time() - start_epoch_time:.2f} seconds")
