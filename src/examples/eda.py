import os
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras
import csv
import numpy as np
from gensim.models import Word2Vec

# Load environment variables from a .env file located in the project root
# load_dotenv() will search for the .env file in the current directory and parent directories
# It returns True if it finds and loads the file, False otherwise.
dotenv_loaded = load_dotenv()

if dotenv_loaded:
    print("Successfully loaded environment variables from .env file.")
else:
    print("Warning: .env file not found. Ensure it exists in the project root or parent directories.")

# Ensure a clean slate by removing the output file if it exists
if os.path.exists("data/processed/processed_rows.csv"):
    os.remove("data/processed/processed_rows.csv")

def get_rows(table_name, limit):
    """
    Fetches a specified number of rows from a PostgreSQL table.

    Args:
        table_name (str): The name of the table to query.
        limit (int): The maximum number of rows to fetch.

    Returns:
        list: A list of rows fetched from the database, represented as DictRow objects.
    """
    # Access database credentials from environment variables
    DB_IP = os.getenv("DB_IP")
    DB_NAME = os.getenv("TABLE_NAME")
    USERNAME = os.getenv("USERNAME")
    PASSWORD = os.getenv("PASSWORD")

    # Establish connection to the PostgreSQL database
    conn_str = f"postgres://{USERNAME}:{PASSWORD}@{DB_IP}/{DB_NAME}"
    conn = psycopg2.connect(conn_str)

    # Use DictCursor to get rows as dictionary-like objects
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Execute the query to fetch 'story' type items
    cur.execute(f"SELECT * FROM hacker_news.{table_name} WHERE type = 'story' LIMIT {limit};")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def sentence_to_vec(sentence, model):
    """
    Converts a sentence into a vector representation using a Word2Vec model.

    Args:
        sentence (str): The input sentence.
        model (Word2Vec): The pre-trained Word2Vec model.

    Returns:
        np.ndarray: The vector representation of the sentence. Returns a zero vector
                    if no words in the sentence are found in the model's vocabulary.
    """
    words = sentence.lower().split()
    # Filter words that exist in the model's vocabulary
    valid_words = [word for word in words if word in model.wv]

    if not valid_words:
        return np.zeros(model.vector_size)  # Fallback for sentences with no known words

    # Get vectors for valid words and compute their mean
    vectors = [model.wv[word] for word in valid_words]
    return np.mean(vectors, axis=0)

def process_rows(rows, debug=False):
    """
    Processes raw database rows into a structured format suitable for analysis.

    Args:
        rows (list): A list of database rows (DictRow objects).
        debug (bool): If True, enables additional debugging output (currently unused).

    Returns:
        list: A list of dictionaries, each representing a processed row.
    """
    # Load the pre-trained Word2Vec model
    # Ensure this path points to the Gensim-native model file, not the PyTorch .pth file.
    model_path = "data/models/word2vec_text8_cbow.model"
    print(f"Loading Gensim Word2Vec model from: {model_path}")
    try:
        model = Word2Vec.load(model_path)
    except FileNotFoundError:
        print(f"Error: Gensim model file not found at {model_path}")
        print("Please ensure the model has been trained using 'upvotes/src/training/train_word2vec_library.py' and saved to the correct location.")
        raise # Re-raise the exception to stop execution
    except Exception as e:
        print(f"Error loading Gensim model: {e}")
        raise # Re-raise other potential loading errors

    processed = []
    print_count = 0  # Counter for printing example rows
    print_limit = 3  # How many example rows to print

    for row in rows:
        # --- Data Cleaning and Filtering ---
        # Keep only 'story' type items
        if row['type'] != 'story':
            continue

        # Skip rows with missing scores
        if row['score'] is None:
            continue

        # Skip rows with missing or empty text
        if row['text'] is None or row['text'] == '':
            continue
        
        # Skip rows with missing or empty titles
        if row['title'] is None or row['title'] == '':
            continue

        # --- Feature Engineering ---
        # Generate vector representations for text and title
        text_vector = sentence_to_vec(row['text'], model)
        text_vector_columns = {f"text_vector_{i}": value for i, value in enumerate(text_vector)}
        title_vector = sentence_to_vec(row['title'], model)
        title_vector_columns = {f"title_vector_{i}": value for i, value in enumerate(title_vector)}

        # Create the processed row dictionary
        processed_row_data = {
            'score': row['score'],
            'dead': row['dead'],
            'text_length': 0 if row['text'] is None else len(row['text'].split(' ')),
            # Potential Bug: title_length uses text length instead of title length
            'title_length': 0 if row['title'] is None else len(row['text'].split(' ')),
            **title_vector_columns,
            **text_vector_columns
        }
        processed.append(processed_row_data)

        # Print the first few processed rows for inspection
        if print_count < print_limit:
            print(f"--- Example Processed Row {print_count + 1} ---")
            # Print only a subset of fields for brevity
            print(f"  Score: {processed_row_data['score']}")
            print(f"  Dead: {processed_row_data['dead']}")
            print(f"  Text Length: {processed_row_data['text_length']}")
            print(f"  Title Length: {processed_row_data['title_length']}")
            # Show first 5 vector components as an example
            title_vec_preview = {k: v for i, (k, v) in enumerate(title_vector_columns.items()) if i < 5}
            text_vec_preview = {k: v for i, (k, v) in enumerate(text_vector_columns.items()) if i < 5}
            print(f"  Title Vector (first 5): {title_vec_preview}")
            print(f"  Text Vector (first 5): {text_vec_preview}")
            print("-" * (len(f"--- Example Processed Row {print_count + 1} ---"))) # Match length of header
            print_count += 1
    
    return processed

# --- Main Execution Logic ---
# Configuration
table_name = "items"  # Source table in the database
num_rows = 50000      # Number of rows to fetch and process

# Fetch data from the database
print(f"Fetching {num_rows} rows from hacker_news.{table_name}...")
rows = get_rows(table_name, num_rows)
print(f"Fetched {len(rows)} rows.")

# Process the fetched rows
print("Processing rows...")
processed_rows = process_rows(rows)
print(f"Processed {len(processed_rows)} rows.")


# Write processed rows to a CSV file
output_csv_path = "data/processed/processed_rows.csv"
print(f"Writing processed rows to {output_csv_path}...")
# Check if any rows were processed before writing
if processed_rows:
    with open(output_csv_path, mode="w", newline="") as csv_file:
        # Use the keys from the first processed row as fieldnames
        writer = csv.DictWriter(csv_file, fieldnames=processed_rows[0].keys())
        writer.writeheader()
        writer.writerows(processed_rows)
    print(f"Successfully wrote processed rows to {output_csv_path}")
else:
    print("No rows were processed, CSV file not written.")

# Final confirmation message is now part of the writing block.

