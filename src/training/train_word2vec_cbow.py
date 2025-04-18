"""
Word2Vec CBOW (Continuous Bag of Words) Implementation
====================================================
This script implements the Word2Vec CBOW model for learning word embeddings.
The CBOW model predicts a target word based on its surrounding context words.
"""

# Import necessary libraries
import csv  # For reading vocabulary files
import multiprocessing  # For determining the number of workers for DataLoader
from datasets import load_dataset  # For loading the text dataset (e.g., text8)
import torch  # PyTorch core library
import torch.nn as nn  # Neural network module (embeddings, linear layers)
import torch.optim as optim  # Optimization algorithms (e.g., Adam)
from torch.utils.data import Dataset, DataLoader  # For creating datasets and loading data in batches
import time  # For timing operations (e.g., epoch duration)
from tqdm import tqdm  # For displaying progress bars during training
import os  # For interacting with the operating system (e.g., setting environment variables)

# Set CUDA_LAUNCH_BLOCKING=1 environment variable.
# This forces CUDA operations to run synchronously, which can make debugging CUDA errors easier
# by providing more precise stack traces, although it might slow down execution slightly.
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Add verbose logging to show what each part of the script is doing and sample data from each stage
# (This comment refers to the print statements added throughout the script for informational purposes)

# Define a function to load the vocabulary mappings from CSV files
def load_tokeniser(token_to_index_filepath, index_to_token_filepath):
    """
    Loads the token-to-index and index-to-token mappings from specified CSV files.

    Args:
        token_to_index_filepath (str): Path to the CSV file containing token-to-index mapping.
                                      Each row should be: token,index
        index_to_token_filepath (str): Path to the CSV file containing index-to-token mapping.
                                      Each row should be: index,token

    Returns:
        tuple: A tuple containing two dictionaries:
               - token_to_index (dict): Maps tokens (words) to their integer indices.
               - index_to_token (dict): Maps integer indices back to their tokens (words).
    """
    print("[INFO] Loading tokeniser...")
    # Initialize an empty dictionary to store the token -> index mapping
    token_to_index = {}
    # Open the token_to_index file for reading
    with open(token_to_index_filepath, 'r') as file:
        # Create a CSV reader object
        reader = csv.reader(file)
        # Iterate over each row in the CSV file
        for row in reader:
            # Unpack the row into token and index
            token, index = row
            # Add the mapping to the dictionary, converting the index to an integer
            token_to_index[token] = int(index)

    # Initialize an empty dictionary to store the index -> token mapping
    index_to_token = {}
    # Open the index_to_token file for reading
    with open(index_to_token_filepath, 'r') as file:
        # Create a CSV reader object
        reader = csv.reader(file)
        # Iterate over each row in the CSV file
        for row in reader:
            # Unpack the row into index and token
            index, token = row
            # Add the mapping to the dictionary, converting the index to an integer
            index_to_token[int(index)] = token

    # Print information about the loaded vocabulary
    print(f"[INFO] Tokeniser loaded with {len(token_to_index)} tokens. Sample: {list(token_to_index.items())[:5]}")
    # Return the two mapping dictionaries
    return token_to_index, index_to_token

# Define a function to load pre-tokenized text (sequence of token indices)
def load_tokenised_text(filepath):
    """
    Loads pre-tokenized text from a file where each line contains space-separated token indices.
    
    Args:
        filepath (str): Path to the file containing tokenized text (indices).
    
    Returns:
        list: A list of integers representing the token indices in the text sequence.
    """
    print("Loading tokenised text...")
    # Open the specified file for reading
    with open(filepath, 'r') as file:
        # Read the entire file content, split it into individual tokens (indices as strings) by space,
        # and convert each token (string) into an integer. Store these integers in a list.
        tokenised_text = [int(token) for token in file.read().split()]
    # Print the number of tokens loaded
    print(f"Tokenised text loaded with {len(tokenised_text)} tokens.")
    # Return the list of token indices
    return tokenised_text

# Define the Word2Vec CBOW model class, inheriting from PyTorch's nn.Module
class Word2VecCBOW(nn.Module):
    """
    Continuous Bag of Words (CBOW) implementation of Word2Vec using PyTorch.
    The model learns word embeddings by predicting a target word based on the
    average of the embeddings of its surrounding context words.
    
    Architecture:
    1. Embedding layer (nn.Embedding): Maps input word indices (context words) to dense vector representations (embeddings).
    2. Linear layer (nn.Linear): Takes the averaged context embedding and projects it to the vocabulary size,
                                  producing scores for each word in the vocabulary as the potential target word.
    """
    def __init__(self, vocab_size, embedding_dim):
        """
        Initialize the CBOW model layers.
        
        Args:
            vocab_size (int): The total number of unique words in the vocabulary.
            embedding_dim (int): The desired dimensionality of the word embeddings (e.g., 100, 200, 300).
        """
        # Call the initializer of the parent class (nn.Module)
        super(Word2VecCBOW, self).__init__()
        print(f"Initializing Word2VecCBOW model with vocab_size={vocab_size}, embedding_dim={embedding_dim}")
        
        # Define the embedding layer.
        # It creates a lookup table where each row corresponds to a word index and contains its embedding vector.
        # Size: vocab_size (number of rows/words) x embedding_dim (size of each vector).
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Define the linear layer (fully connected layer).
        # It takes the averaged context embedding (size: embedding_dim) as input
        # and outputs scores for each word in the vocabulary (size: vocab_size).
        self.linear = nn.Linear(embedding_dim, vocab_size)

    # Define the forward pass of the model
    def forward(self, context):
        """
        Defines how the model processes input context indices to produce output scores.
        
        Args:
            context (torch.Tensor): A tensor containing batches of context word indices.
                                    Shape: (batch_size, context_window_size * 2)
                                    
        Returns:
            torch.Tensor: A tensor containing output scores for each word in the vocabulary for each sample in the batch.
                          Shape: (batch_size, vocab_size)
        """
        try:
            # --- Input Validation ---
            # Check if any context index is out of the valid range [0, vocab_size - 1].
            # self.embeddings.num_embeddings is equivalent to vocab_size.
            if torch.any(context >= self.embeddings.num_embeddings) or torch.any(context < 0):
                print(f"Invalid context indices detected: min={context.min()}, max={context.max()}, vocab_size={self.embeddings.num_embeddings}")
                # Clamp the indices to the valid range to prevent errors.
                # This replaces invalid indices with the closest valid index (0 or vocab_size - 1).
                context = torch.clamp(context, 0, self.embeddings.num_embeddings - 1)

            # --- Embedding Lookup ---
            # Pass the context indices through the embedding layer.
            # This retrieves the corresponding embedding vectors for each context word.
            # Input context shape: (batch_size, context_window_size * 2)
            # Output embedded shape: (batch_size, context_window_size * 2, embedding_dim)
            embedded = self.embeddings(context)
            #print(f"Embedded shape before mean: {embedded.shape}") # Debug print

            # --- Averaging Context Embeddings ---
            # Calculate the mean of the embedding vectors for the context words in each sample.
            # The mean is taken along dimension 1 (the dimension representing the different context words).
            # Input embedded shape: (batch_size, context_window_size * 2, embedding_dim)
            # Output embedded shape: (batch_size, embedding_dim)
            embedded = torch.mean(embedded, dim=1)
            #print(f"Embedded shape after mean: {embedded.shape}") # Debug print

            # --- Linear Projection ---
            # Pass the averaged context embedding through the linear layer.
            # This projects the embedding onto the vocabulary space, generating scores for each word.
            # Input embedded shape: (batch_size, embedding_dim)
            # Output out shape: (batch_size, vocab_size)
            out = self.linear(embedded)
            
            # Return the final output scores
            return out

        # --- Error Handling ---
        # Catch any exceptions during the forward pass for easier debugging.
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            # Print shapes of relevant tensors if they exist, aiding in diagnosing shape mismatches.
            print(f"Context shape: {context.shape if 'context' in locals() else 'Not available'}")
            print(f"Embedded shape: {embedded.shape if 'embedded' in locals() else 'Not available'}")
            # Re-raise the exception to halt execution if an error occurs.
            raise

# Define a custom PyTorch Dataset class for CBOW
class CBOWDataset(Dataset):
    """
    Custom Dataset class for generating CBOW (context, target) pairs.
    Inherits from torch.utils.data.Dataset, requiring implementation of __len__ and __getitem__.
    """
    def __init__(self, text, context_size, token_to_index):
        """
        Initialize the CBOW dataset.
        
        Args:
            text (list): The input corpus as a list of word strings.
            context_size (int): The number of words to consider on each side of the target word (window radius).
            token_to_index (dict): The dictionary mapping words (tokens) to their integer indices.
        """
        print(f"Initializing CBOWDataset with context_size={context_size}")
        # Store the input text (list of words)
        self.text = text
        # Store the context window size (radius)
        self.context_size = context_size
        # Store the token-to-index mapping dictionary
        self.token_to_index = token_to_index
        # Get the index for the <UNK> (unknown) token, defaulting to 0 if not found.
        self.unk = token_to_index.get("<UNK>", 0)
        # Determine the vocabulary size from the mapping dictionary
        self.vocab_size = len(token_to_index)
        print(f"Vocabulary size: {self.vocab_size}")
        # Create the actual training data (list of (context_indices, target_index) pairs)
        # by calling the create_cbow_data method.
        self.data = self.create_cbow_data()

    def create_cbow_data(self):
        # This method is intentionally left uncommented as per the user's request.
        # Its comments were added in the previous step.
        """
        Generates context-target pairs for CBOW training.
        Iterates through the text and for each word (target), it gathers the surrounding words (context).
        """
        # Initialize an empty list to store the (context, target) pairs.
        data = []
        print("[INFO] Creating CBOW data...")
        # Use tqdm for a progress bar during data creation.
        with tqdm(total=len(self.text), desc="Processing CBOW data") as pbar:
            # Iterate through the text, starting from the first possible target word
            # (index = context_size) up to the last possible target word
            # (index = len(text) - context_size - 1).
            for i in range(self.context_size, len(self.text) - self.context_size):
                # Extract the context words:
                # - Words before the target: self.text[i - self.context_size : i]
                # - Words after the target: self.text[i + 1 : i + self.context_size + 1]
                # Convert each context word to its index using token_to_index.
                # If a word is not found, use the index for the <UNK> token.
                context_words = self.text[i - self.context_size:i] + self.text[i + 1:i + self.context_size + 1]
                context = [self.token_to_index.get(token, self.unk) for token in context_words]
                
                # Get the target word (the word at the current index i).
                # Convert the target word to its index using token_to_index.
                # If the target word is not found, use the index for the <UNK> token.
                target = self.token_to_index.get(self.text[i], self.unk)
                
                # Validate that all context indices and the target index are within the valid range [0, vocab_size - 1].
                # This step prevents out-of-bounds errors when creating tensors later.
                context = [max(0, min(idx, self.vocab_size - 1)) for idx in context]
                target = max(0, min(target, self.vocab_size - 1))
                
                # Check if any index is still invalid after clamping (should not happen with the clamping above, but as a safeguard).
                if any(idx >= self.vocab_size or idx < 0 for idx in context) or target >= self.vocab_size or target < 0:
                    print(f"Invalid index found after clamping: context={context}, target={target}, vocab_size={self.vocab_size}")
                    # Skip this sample if indices are invalid.
                    continue
                    
                # Ensure the context list has the correct number of words (2 * context_size)
                # and the target is valid (not None).
                if len(context) == 2 * self.context_size and target is not None:
                    # Append the (context, target) pair to the data list.
                    data.append((context, target))

                # Log progress periodically (e.g., every 100,000 samples).
                # This helps monitor the data creation process, especially for large datasets.
                if len(data) % 100000 == 0 and len(data) > 0:  # Log every 100,000 samples, avoid logging at 0
                    print(f"[DEBUG] Processed {len(data)} samples. Last context: {context}, Target: {target}")

                # Update the progress bar.
                pbar.update(1)
        # Print a message indicating completion and the total number of samples created.
        print(f"[INFO] CBOW data creation complete. Total samples: {len(data)}")
        # Return the list of (context, target) pairs.
        return data

    # Define the __len__ method, required by PyTorch Dataset
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        Called by DataLoader to determine the size of the dataset.
        """
        # Return the length of the pre-generated data list
        return len(self.data)

    # Define the __getitem__ method, required by PyTorch Dataset
    def __getitem__(self, idx):
        """
        Retrieves a single training sample (context and target) at the specified index.
        Called by DataLoader to fetch individual samples for batching.
        
        Args:
            idx (int): The index of the desired sample in the self.data list.
            
        Returns:
            tuple: A tuple containing two tensors:
                   - context_tensor (torch.LongTensor): Tensor of context word indices. Shape: (context_size * 2,)
                   - target_tensor (torch.LongTensor): Tensor containing the single target word index. Shape: () (scalar)
        """
        # Retrieve the context list (integers) and target integer from the pre-generated data
        context, target = self.data[idx]
        
        # --- Tensor Conversion ---
        # Ensure context indices are integers (might be redundant if create_cbow_data guarantees ints, but safe)
        context = [int(x) for x in context]
        # Ensure target index is an integer
        target = int(target)
        
        # Convert the list of context indices into a PyTorch tensor of type Long (int64).
        # .view(-1) ensures it's a 1D tensor, although it likely already is.
        context_tensor = torch.tensor(context, dtype=torch.long).view(-1)
        # Convert the target index into a PyTorch scalar tensor of type Long (int64).
        target_tensor = torch.tensor(target, dtype=torch.long)
        
        # Return the context and target tensors as a tuple
        return context_tensor, target_tensor

# --- Main Execution Block ---
# This code runs only when the script is executed directly (not imported as a module)
if __name__ == '__main__':
    print("[INFO] Starting training script...")

    # --- 1. Load Vocabulary ---
    print("[INFO] Loading tokens...")
    # Define filepaths for the vocabulary mapping files
    token_to_index_filepath = 'data/processed/token_to_index.csv'
    index_to_token_filepath = 'data/processed/index_to_token.csv'
    # Load the mappings using the defined function
    token_to_index, index_to_token = load_tokeniser(token_to_index_filepath, index_to_token_filepath)
    # Determine the vocabulary size from the loaded mapping
    vocab_size = len(token_to_index)
    
    # --- 2. Load and Prepare Dataset ---
    print("[INFO] Loading dataset...")
    # Load the text8 dataset from the 'datasets' library (Hugging Face)
    # If not available locally, it might download it.
    # Note: Using pre-tokenized text might be more efficient if available via load_tokenised_text.
    # This example uses the raw text from the dataset.
    dataset = load_dataset("afmck/text8") 
    # Extract the raw text string from the dataset (assuming 'train' split, first element, 'text' field)
    text = dataset['train'][0]['text'] 
    # Split the raw text into a list of word strings
    text_words = text.split()
    print(f"[INFO] Loaded text with {len(text_words)} tokens. Sample: {text_words[:10]}")

    print("[INFO] Preparing CBOW dataset...")
    # Define the context window size (radius)
    context_size = 2  # Looks at 2 words before and 2 words after the target
    # Create an instance of the CBOWDataset
    cbow_dataset = CBOWDataset(text_words, context_size, token_to_index)
    
    # --- 3. Create DataLoader ---
    # Determine the number of worker processes for loading data in parallel.
    # Uses a maximum of 4, or the number of CPU cores available, whichever is smaller.
    num_workers = min(4, multiprocessing.cpu_count())
    # Define the batch size for training
    batch_size = 128
    # Create a DataLoader instance.
    # - cbow_dataset: The dataset to load from.
    # - batch_size: Number of samples per batch.
    # - shuffle=True: Randomly shuffle the data at the beginning of each epoch.
    # - num_workers: Number of subprocesses to use for data loading.
    # - pin_memory=True: If using CUDA, this can speed up data transfer to the GPU.
    dataloader = DataLoader(cbow_dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=num_workers, pin_memory=True)
    print(f"[INFO] Created DataLoader with {len(dataloader)} batches of size {batch_size}")

    # --- 4. Initialize Model and Training Components ---
    print("[INFO] Initializing model...")
    # Define the dimensionality for the word embeddings
    embedding_dim = 200
    # Create an instance of the Word2VecCBOW model
    model = Word2VecCBOW(vocab_size, embedding_dim)

    # --- 5. Set Up Device (CPU/GPU) and Move Model ---
    # Check if CUDA (GPU support) is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    # Move the model's parameters and buffers to the selected device (GPU or CPU)
    model.to(device)
    
    # Define the loss function: CrossEntropyLoss is suitable for multi-class classification
    # (predicting the target word index out of the entire vocabulary).
    # It combines LogSoftmax and NLLLoss. Move the criterion to the selected device.
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Define the optimizer: Adam is a popular choice for optimizing neural networks.
    # It adapts the learning rate for each parameter. Pass the model's parameters and learning rate.
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- 6. Training Loop ---
    print("[INFO] Training model...")
    # Define the number of training epochs (passes over the entire dataset)
    num_epochs = 5
    
    # Initialize a gradient scaler for mixed precision training if CUDA is available.
    # Mixed precision uses float16 for some computations to speed up training and reduce memory usage,
    # while maintaining numerical stability using float32 where needed.
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None # Corrected from torch.amp.GradScaler
    
    # Loop over the specified number of epochs
    for epoch in range(num_epochs):
        print(f"[INFO] Starting epoch {epoch + 1}/{num_epochs}...")
        # Initialize the total loss for this epoch
        epoch_loss = 0
        # Initialize batch counter
        i = 0
        # Record the start time of the epoch
        start_epoch_time = time.time()
        
        # Use tqdm to create a progress bar for the dataloader iteration
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            # Iterate over batches provided by the DataLoader
            for context, target in dataloader:
                # Record the start time for processing this batch (for debugging/profiling)
                start_sample_time = time.time()
                
                # --- Move Data to Device ---
                # Move the context and target tensors to the selected device (GPU/CPU).
                # non_blocking=True can potentially speed up transfer if pin_memory=True was used in DataLoader.
                context, target = context.to(device, non_blocking=True), target.to(device, non_blocking=True)
                
                # --- Input Validation (Target) ---
                # Optional: Validate target indices are within vocabulary range before passing to loss function.
                # This helps catch potential errors earlier.
                if torch.any(target >= vocab_size) or torch.any(target < 0):
                    print(f"Invalid target indices detected: min={target.min()}, max={target.max()}, vocab_size={vocab_size}")
                    # Decide how to handle this - skip batch, clamp, etc. Here we just print.
                
                # --- Forward and Backward Pass ---
                # Reset gradients from the previous iteration
                optimizer.zero_grad()
                
                # Check if mixed precision training is enabled (scaler exists)
                if scaler:
                    # Use autocast to automatically handle mixed precision operations
                    with torch.cuda.amp.autocast(): # Corrected from torch.amp.autocast
                        # Forward pass: Get model predictions for the context
                        output = model(context)
                        # Calculate loss between predictions (output) and actual targets
                        loss = criterion(output, target)
                    # Backward pass (gradient calculation) using the scaler
                    scaler.scale(loss).backward()
                    # Update model parameters using the optimizer, managed by the scaler
                    scaler.step(optimizer)
                    # Update the scaler for the next iteration
                    scaler.update()
                else:  # Standard precision training (no scaler)
                    # Forward pass
                    output = model(context)
                    # Calculate loss
                    loss = criterion(output, target)
                    # Backward pass (gradient calculation)
                    loss.backward()
                    # Update model parameters
                    optimizer.step()
                
                # --- Track Progress ---
                # Add the loss of the current batch to the total epoch loss
                # .item() gets the scalar value from the loss tensor
                epoch_loss += loss.item()
                
                # Update progress bar periodically (e.g., every 1000 batches)
                if i % 1000 == 0:
                    pbar.update(1000) # Update progress bar steps
                    # Set postfix information on the progress bar (current batch loss, time per sample)
                    pbar.set_postfix(loss=loss.item(), 
                                   sample_time=f"{(time.time() - start_sample_time) * 1000:.2f}ms")
                # Log detailed debug information less frequently (e.g., every 100 batches)
                if i % 100 == 0:
                    print(f"[DEBUG] Batch {i}, Loss: {loss.item():.4f}, Sample context: {context[0].tolist()}, Target: {target[0].item()}")
                
                # Increment batch counter
                i += 1
        
        # --- End of Epoch ---
        # Calculate the average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
        # Print epoch summary statistics
        print(f"[INFO] Epoch {epoch + 1} complete. Average Loss: {avg_loss:.4f}, "
              f"Time: {time.time() - start_epoch_time:.2f} seconds")

    # --- 7. Save Trained Model ---
    print("[INFO] Saving model and optimizer state...")
    # Define the path where the model checkpoint will be saved
    save_path = 'data/models/word2vec_cbow.pth'
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure directory exists
    # Save the model's state dictionary (weights and biases) and the optimizer's state
    # This allows resuming training or using the model for inference later.
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        # You could also save other things like epoch number, loss, vocab mapping etc.
    }, save_path)
    print(f"[INFO] Model and optimizer state saved to {save_path}")