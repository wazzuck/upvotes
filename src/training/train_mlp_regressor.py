import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import logging
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Define paths relative to the script or project root
# Assumes script is run from the project root `upvotes/`
PROCESSED_DATA_DIR = Path("data/processed/")
MODEL_DIR = Path("models/")
MODEL_FILENAME = "mlp_regressor_model.pkl"
SCALER_FILENAME = "scaler.pkl"

# --- Helper Functions ---
def load_data(file_path: Path):
    """Loads data from parquet or csv."""
    logger.info(f"Loading data from {file_path}...")
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

def save_artifact(artifact, file_path: Path):
    """Saves a Python object using joblib."""
    logger.info(f"Saving artifact to {file_path}...")
    file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    joblib.dump(artifact, file_path)
    logger.info(f"Artifact saved successfully.")

# --- Main Training Logic ---
def train_model(data_path: Path, target_column: str, feature_columns: list):
    """Loads data, trains an MLP regressor, evaluates, and saves artifacts."""
    df = load_data(data_path)

    # --- Feature Selection ---
    # Ensure target and features exist in the dataframe
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found in data: {missing_features}")

    X = df[feature_columns]
    y = df[target_column]

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # --- Data Splitting ---
    logger.info("Splitting data into training and validation sets (80/20 split)...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Train set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")

    # --- Model Pipeline ---
    # Create a pipeline including scaling and the MLP regressor
    # Scaling is often crucial for MLP performance
    logger.info("Defining model pipeline (StandardScaler + MLPRegressor)...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()), # Scale features
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(100, 50), # Example: 2 hidden layers with 100 and 50 neurons
            activation='relu',        # Rectified Linear Unit activation function
            solver='adam',            # Optimizer
            alpha=0.0001,             # L2 penalty (regularization term) parameter
            batch_size='auto',        # Size of minibatches for stochastic optimizers
            learning_rate='constant', # Learning rate schedule for weight updates
            learning_rate_init=0.001, # Initial learning rate
            max_iter=200,             # Maximum number of iterations
            shuffle=True,             # Whether to shuffle samples in each iteration
            random_state=42,          # For reproducibility
            early_stopping=True,      # Stop training when validation score is not improving
            validation_fraction=0.1,  # Proportion of training data to set aside as validation set for early stopping
            n_iter_no_change=10,      # Number of iterations with no improvement to wait before stopping
            verbose=True              # Print progress messages
        ))
    ])

    # --- Model Training ---
    logger.info("Training the model pipeline...")
    pipeline.fit(X_train, y_train)
    logger.info("Model training complete.")

    # --- Evaluation ---
    logger.info("Evaluating model on the validation set...")
    y_pred_val = pipeline.predict(X_val)
    mse = mean_squared_error(y_val, y_pred_val)
    rmse = mse**0.5
    r2 = r2_score(y_val, y_pred_val)
    logger.info(f"Validation Mean Squared Error (MSE): {mse:.4f}")
    logger.info(f"Validation Root Mean Squared Error (RMSE): {rmse:.4f}")
    logger.info(f"Validation R-squared (R2): {r2:.4f}")

    # --- Save Artifacts ---
    model_path = MODEL_DIR / MODEL_FILENAME
    scaler_path = MODEL_DIR / SCALER_FILENAME

    # Save the entire pipeline (includes scaler and model)
    save_artifact(pipeline, model_path)

    # Optional: Save the scaler separately if needed elsewhere, though it's in the pipeline
    # scaler = pipeline.named_steps['scaler']
    # save_artifact(scaler, scaler_path)

    logger.info("Training script finished.")

# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP Regressor model for Hacker News Upvotes.")
    parser.add_argument(
        "--data-file",
        type=str,
        default="features.parquet", # Default filename in the processed data dir
        help="Filename of the processed data file (e.g., features.parquet or features.csv)"
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="score", # Assuming 'score' is the upvote count column
        help="Name of the target variable column (upvotes/score)"
    )
    # Add more arguments if needed, e.g., for feature list, hyperparameters

    args = parser.parse_args()

    # Construct full data path
    data_full_path = PROCESSED_DATA_DIR / args.data_file

    # --- Define Feature Columns ---
    # !!! IMPORTANT: Replace this list with the actual column names of your features
    # This should include names for title embeddings (e.g., title_emb_0, title_emb_1, ...),
    # user features, time features, domain features, etc.
    feature_cols = [
        # Example features - REPLACE THESE
        'title_emb_0', 'title_emb_1', # ..., 'title_emb_99'
        'author_karma',
        'author_account_age_days',
        'post_hour',
        'post_day_of_week',
        # 'domain_encoded',
        # 'title_length'
    ]
    # You might need to generate the embedding column names dynamically
    # feature_cols = [f'title_emb_{i}' for i in range(100)] + ['author_karma', ...]

    train_model(data_full_path, args.target_column, feature_cols) 