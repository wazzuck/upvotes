import pandas as pd
import numpy as np
import gensim
from pathlib import Path
import logging
import argparse
import re
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional: Setup tqdm for pandas progress_apply
tqdm.pandas()

# --- Configuration via Arguments ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate features including Word2Vec title embeddings.")
    parser.add_argument(
        "--input-data",
        type=str,
        default="data/processed/processed_items_users.parquet",
        help="Path to the input processed data file (e.g., containing titles and other features)."
    )
    parser.add_argument(
        "--w2v-model",
        type=str,
        default="models/word2vec.model",
        help="Path to the trained Word2Vec model file."
    )
    parser.add_argument(
        "--output-data",
        type=str,
        default="data/processed/features.parquet",
        help="Path to save the final features file (with embeddings)."
    )
    parser.add_argument(
        "--title-column",
        type=str,
        default="title",
        help="Name of the column containing the post titles."
    )
    # Add arguments for other features if needed for specific processing
    return parser.parse_args()

# --- Helper Functions ---
def load_data(file_path: Path):
    """Loads data from parquet or csv."""
    logger.info(f"Loading data from {file_path}...")
    if not file_path.exists():
        logger.error(f"Input data file not found: {file_path}")
        raise FileNotFoundError(f"Input data file not found: {file_path}")

    if file_path.suffix == '.parquet':
        try:
            return pd.read_parquet(file_path)
        except ImportError:
            logger.error("pyarrow or fastparquet needed to read parquet files. pip install pyarrow or pip install fastparquet")
            raise
    elif file_path.suffix == '.csv':
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def tokenize(text: str):
    """Simple whitespace tokenizer, lowercases, and removes basic punctuation."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    # Remove punctuation but keep spaces and word characters
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def get_average_embedding(title: str, w2v_model: gensim.models.Word2Vec, vector_size: int):
    """Calculates the average Word2Vec embedding for a title."""
    tokens = tokenize(title)
    # Use get method with default to handle out-of-vocabulary words gracefully
    known_vectors = [w2v_model.wv.get_vector(token) for token in tokens if w2v_model.wv.has_index_for(token)]

    if not known_vectors:
        # Return zeros if no known words found or title is empty
        return np.zeros(vector_size, dtype=np.float32)

    # Calculate mean of the vectors
    return np.mean(known_vectors, axis=0, dtype=np.float32)

def save_data(df: pd.DataFrame, file_path: Path):
    """Saves DataFrame to parquet or csv."""
    logger.info(f"Saving final features to {file_path}...")
    file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    if file_path.suffix == '.parquet':
        try:
            df.to_parquet(file_path, index=False)
        except ImportError:
            logger.error("pyarrow or fastparquet needed to save parquet files. pip install pyarrow or pip install fastparquet")
            raise
    elif file_path.suffix == '.csv':
        df.to_csv(file_path, index=False)
    else:
        raise ValueError(f"Unsupported file format for saving: {file_path.suffix}")
    logger.info(f"Features saved successfully to {file_path}")

# --- Main Feature Generation Logic ---
def main():
    args = parse_arguments()

    input_data_path = Path(args.input_data)
    w2v_model_path = Path(args.w2v_model)
    output_data_path = Path(args.output_data)
    title_col = args.title_column

    # --- Load Data and Model ---
    df = load_data(input_data_path)

    logger.info(f"Loading Word2Vec model from {w2v_model_path}...")
    if not w2v_model_path.exists():
        logger.error(f"Word2Vec model file not found: {w2v_model_path}")
        raise FileNotFoundError(f"Word2Vec model file not found: {w2v_model_path}")
    try:
        # Load the model (adjust if using KeyedVectors directly)
        w2v_model = gensim.models.Word2Vec.load(str(w2v_model_path))
        vector_size = w2v_model.vector_size
        logger.info(f"Word2Vec model loaded successfully. Vector size: {vector_size}")
    except Exception as e:
        logger.error(f"Failed to load Word2Vec model: {e}")
        raise

    # --- Check Title Column ---
    if title_col not in df.columns:
        logger.error(f"Title column '{title_col}' not found in the input data.")
        raise ValueError(f"Title column '{title_col}' not found in the input data.")
    # Handle potential NaN values in title column safely
    df[title_col] = df[title_col].fillna('').astype(str)

    # --- Generate Embeddings ---
    logger.info(f"Generating average Word2Vec embeddings for column '{title_col}'...")
    # Using progress_apply from tqdm for visual feedback
    # Ensure get_average_embedding handles potential errors if w2v_model parts are missing
    embeddings = df[title_col].progress_apply(get_average_embedding, w2v_model=w2v_model, vector_size=vector_size)
    logger.info("Embeddings generated.")

    # --- Combine Features ---
    logger.info("Combining embeddings with original features...")
    # Create column names for embeddings
    embedding_cols = [f'title_emb_{i}' for i in range(vector_size)]
    # Create a DataFrame from the embeddings Series (which contains numpy arrays)
    embeddings_df = pd.DataFrame(embeddings.to_list(), columns=embedding_cols, index=df.index)

    # Concatenate original df with embeddings df
    final_df = pd.concat([df, embeddings_df], axis=1)

    # Optional: Drop the original title column if no longer needed for the model
    # final_df = final_df.drop(columns=[title_col])
    # logger.info(f"Dropped original title column: '{title_col}'")

    logger.info(f"Final features DataFrame shape: {final_df.shape}")
    logger.info(f"Columns in final DataFrame: {final_df.columns.tolist()}")


    # --- Save Final Features ---
    save_data(final_df, output_data_path)

if __name__ == "__main__":
    main() 