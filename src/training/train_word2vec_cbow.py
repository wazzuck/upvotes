"""
Word2Vec CBOW (Continuous Bag of Words) Implementation
====================================================
This script implements the Word2Vec CBOW model for learning word embeddings.
The CBOW model predicts a target word based on its surrounding context words.
"""

import csv
import multiprocessing
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
import os

# Set CUDA_LAUNCH_BLOCKING for synchronous execution to help debug CUDA errors
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def load_tokeniser(token_to_index_filepath, index_to_token_filepath):
    """
    Loads the token-to-index and index-to-token mappings from CSV files.
    These mappings are used to convert between words and their numerical representations.
    
    Args:
        token_to_index_filepath: Path to CSV file mapping words to indices
        index_to_token_filepath: Path to CSV file mapping indices to words
    
    Returns:
        tuple: (token_to_index dict, index_to_token dict)
    """
    print("Loading tokeniser...")
    token_to_index = {}
    with open(token_to_index_filepath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            token, index = row
            token_to_index[token] = int(index)

    index_to_token = {}
    with open(index_to_token_filepath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            index, token = row
            index_to_token[int(index)] = token

    print(f"Tokeniser loaded with {len(token_to_index)} tokens.")
    return token_to_index, index_to_token

def load_tokenised_text(filepath):
    """
    Loads pre-tokenized text from a file where each line contains a token index.
    
    Args:
        filepath: Path to the file containing tokenized text
    
    Returns:
        list: List of token indices
    """
    print("Loading tokenised text...")
    with open(filepath, 'r') as file:
        tokenised_text = [int(token) for token in file.read().split()]
    print(f"Tokenised text loaded with {len(tokenised_text)} tokens.")
    return tokenised_text

class Word2VecCBOW(nn.Module):
    """
    Continuous Bag of Words (CBOW) implementation of Word2Vec.
    The model learns word embeddings by predicting a target word from its context.
    
    Architecture:
    1. Embedding layer: Maps word indices to dense vectors
    2. Linear layer: Projects embeddings to vocabulary size for prediction
    """
    def __init__(self, vocab_size, embedding_dim):
        """
        Initialize the CBOW model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
        """
        super(Word2VecCBOW, self).__init__()
        print(f"Initializing Word2VecCBOW model with vocab_size={vocab_size}, embedding_dim={embedding_dim}")
        # Embedding layer: converts word indices to dense vectors
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Linear layer: projects embeddings to vocabulary size for prediction
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        """
        Forward pass of the CBOW model.
        
        Process:
        1. Convert context word indices to embeddings
        2. Average the context embeddings
        3. Project to vocabulary size for prediction
        
        Args:
            context: Tensor of shape [batch_size, context_size] containing word indices
        
        Returns:
            Tensor: Logits for predicting the target word
        """
        try:
            # Validate context indices are within vocabulary range
            if torch.any(context >= self.embeddings.num_embeddings):
                print(f"Invalid index found: max index {context.max()}, vocab size {self.embeddings.num_embeddings}")
                context = torch.clamp(context, 0, self.embeddings.num_embeddings - 1)
            
            # Print min and max context indices for debugging
            print(f"Context indices range: min={context.min()}, max={context.max()}")
            
            print(f"Context shape: {context.shape}")
            
            # Convert word indices to embeddings
            embedded = self.embeddings(context)
            print(f"Embedded shape before mean: {embedded.shape}")
            
            # Ensure proper tensor dimensions
            if embedded.dim() == 2:
                embedded = embedded.unsqueeze(0)
            
            print(f"Embedded shape before mean: {embedded.shape}")
            
            # Average context embeddings
            embedded = torch.mean(embedded, dim=2)
            print(f"Embedded shape after mean: {embedded.shape}")
            
            # Project to vocabulary size for prediction
            out = self.linear(embedded)
            return out
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Context shape: {context.shape if 'context' in locals() else 'Not available'}")
            print(f"Embedded shape: {embedded.shape if 'embedded' in locals() else 'Not available'}")
            raise

class CBOWDataset(Dataset):
    """
    Custom Dataset class for CBOW training.
    Creates context-target pairs from the input text.
    """
    def __init__(self, text, context_size, token_to_index):
        """
        Initialize the CBOW dataset.
        
        Args:
            text: List of words
            context_size: Number of words to consider on each side of target
            token_to_index: Dictionary mapping words to indices
        """
        print(f"Initializing CBOWDataset with context_size={context_size}")
        self.text = text
        self.context_size = context_size
        self.token_to_index = token_to_index
        self.unk = token_to_index.get("<UNK>", 0)  # Default to 0 if UNK not found
        self.vocab_size = len(token_to_index)
        print(f"Vocabulary size: {self.vocab_size}")
        self.data = self.create_cbow_data()

    def create_cbow_data(self):
        """
        Creates context-target pairs for CBOW training.
        For each word in the text, creates a pair of:
        - Context: surrounding words
        - Target: the word to predict
        
        Returns:
            list: List of (context, target) pairs
        """
        data = []
        print("Creating CBOW data...")
        i = 0
        with tqdm(total=len(self.text), desc="Processing CBOW data") as pbar:
            for i in range(self.context_size, len(self.text) - self.context_size):
                # Get context words (before and after target)
                context = [self.token_to_index.get(token, self.unk) for token in 
                         self.text[i - self.context_size:i] + 
                         self.text[i + 1:i + self.context_size + 1]]
                target = self.token_to_index.get(self.text[i], self.unk)
                
                # Validate indices
                if any(idx >= self.vocab_size for idx in context) or target >= self.vocab_size:
                    print(f"Invalid index found: context {context}, target {target}, vocab size {self.vocab_size}")
                    continue
                    
                if len(context) == 2 * self.context_size and target is not None:
                    data.append((context, target))
                if i % 1000 == 0:
                    pbar.update(1000)
        print(f"CBOW data creation complete. Total samples: {len(data)}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single training sample.
        
        Args:
            idx: Index of the sample to return
            
        Returns:
            tuple: (context_tensor, target_tensor)
        """
        context, target = self.data[idx]
        # Convert to tensors
        context = [int(x) for x in context]
        target = int(target)
        context_tensor = torch.tensor(context, dtype=torch.long).view(-1)
        target_tensor = torch.tensor(target, dtype=torch.long)
        return context_tensor, target_tensor

if __name__ == '__main__':
    # 1. Load vocabulary mappings
    print("Loading tokens...")
    token_to_index_filepath = 'data/processed/token_to_index.csv'
    index_to_token_filepath = 'data/processed/index_to_token.csv'
    token_to_index, index_to_token = load_tokeniser(token_to_index_filepath, index_to_token_filepath)
    
    # 2. Load and prepare dataset
    print("Loading dataset...")
    dataset = load_dataset("afmck/text8")
    text = dataset['train'][0]['text']
    print(f"Loaded text with {len(text.split())} tokens")

    print("Preparing CBOW dataset...")
    context_size = 2  # Number of words to consider on each side
    cbow_dataset = CBOWDataset(text.split(), context_size, token_to_index)
    
    # 3. Create DataLoader for efficient batching
    num_workers = min(4, multiprocessing.cpu_count())
    dataloader = DataLoader(cbow_dataset, batch_size=128, shuffle=True, 
                          num_workers=num_workers, pin_memory=True)
    print(f"Created DataLoader with {len(dataloader)} batches of size 128")

    # 4. Initialize model and training components
    print("Initializing model...")
    vocab_size = len(token_to_index)
    embedding_dim = 200

    # 5. Set up device and move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = Word2VecCBOW(vocab_size, embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. Training loop
    print("Training model...")
    num_epochs = 5
    # Use mixed precision training if GPU is available
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        i = 0
        start_epoch_time = time.time()
        
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for context, target in dataloader:
                start_sample_time = time.time()
                
                # Move data to device
                context, target = context.to(device, non_blocking=True), target.to(device, non_blocking=True)
                
                # Forward pass
                optimizer.zero_grad()
                if scaler:  # Mixed precision training
                    with torch.amp.autocast(device_type='cuda'):
                        output = model(context)
                        loss = criterion(output, target)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:  # Standard training
                    output = model(context)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                
                # Track progress
                epoch_loss += loss.item()
                if i % 1000 == 0:
                    pbar.update(1000)
                    pbar.set_postfix(loss=loss.item(), 
                                   sample_time=f"{(time.time() - start_sample_time) * 1000:.2f}ms")
                i += 1
        
        # Print epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Avg Loss: {avg_loss:.4f}, "
              f"Time: {time.time() - start_epoch_time:.2f} seconds")

    # 7. Save trained model
    print("Saving model and optimizer state...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, 'data/models/word2vec_cbow.pth')
    print("Model and optimizer state saved.")