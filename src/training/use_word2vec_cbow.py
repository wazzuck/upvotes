import torch
from torch import nn
import csv
from numpy import dot
from numpy.linalg import norm
import argparse

# Define the Word2VecCBOW class (same as in training)
class Word2VecCBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecCBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        if torch.any(context >= self.embeddings.num_embeddings) or torch.any(context < 0):
            raise ValueError(f"Invalid context indices: {context}")
        embedded = self.embeddings(context)
        embedded = torch.mean(embedded, dim=1)  # Average context embeddings
        out = self.linear(embedded)
        return out

# Load the model
def load_model(model_path, vocab_size, embedding_dim, device):
    model = Word2VecCBOW(vocab_size, embedding_dim)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Load the tokeniser
def load_tokeniser(token_to_index_filepath, index_to_token_filepath):
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

    return token_to_index, index_to_token

# Get word embedding
def get_word_embedding(word, model, token_to_index, device):
    word_index = token_to_index.get(word, token_to_index.get("<UNK>", 0))
    word_tensor = torch.tensor([word_index], dtype=torch.long).to(device)
    embedding = model.embeddings(word_tensor).detach().cpu().numpy()
    return embedding

# Calculate cosine similarity
def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# Predict target word from context
def predict_target(context_words, model, token_to_index, index_to_token, device):
    # Convert words to indices
    context_indices = [token_to_index.get(word.lower(), token_to_index.get("<UNK>", 0)) for word in context_words]
    print(f"Context indices: {context_indices}")

    # Clamp indices to valid range
    context_indices = [max(0, min(idx, model.embeddings.num_embeddings - 1)) for idx in context_indices]

    # Create tensor
    context_tensor = torch.tensor(context_indices, dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension

    # Forward pass
    with torch.no_grad():
        output = model(context_tensor)
        predicted_index = torch.argmax(output, dim=1).item()

    return index_to_token[predicted_index]

# Main function
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict the missing word in a context.")
    parser.add_argument("context", type=str, help="A set of words in quotes, e.g., 'this is an example'")
    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model and tokeniser paths
    model_path = '/root/upvotes/data/models/word2vec_cbow.pth'
    token_to_index_filepath = '/root/upvotes/data/processed/token_to_index.csv'
    index_to_token_filepath = '/root/upvotes/data/processed/index_to_token.csv'

    # Model parameters
    vocab_size = 67428  # Replace with your actual vocab size
    embedding_dim = 200  # Replace with your actual embedding dimension

    # Load model and tokeniser
    model = load_model(model_path, vocab_size, embedding_dim, device)
    print("Model loaded successfully.")
    token_to_index, index_to_token = load_tokeniser(token_to_index_filepath, index_to_token_filepath)
    print("Tokeniser loaded successfully.")

    # Get context words from command-line input
    context_words = args.context.split()
    print(f"Context words: {context_words}")

    # Predict the missing word
    predicted_word = predict_target(context_words, model, token_to_index, index_to_token, device)
    print(f"Predicted word for context '{context_words}': {predicted_word}")