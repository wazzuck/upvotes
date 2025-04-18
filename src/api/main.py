import os
import joblib  # Or pickle, or whatever you used to save the model
# import gensim # If using gensim for Word2Vec
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Use environment variables for model paths for flexibility
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/")) # Default to models/ subdir relative to where app is run
MODEL_PATH = MODEL_DIR / os.getenv("MODEL_FILENAME", "model.pkl")
W2V_MODEL_PATH = MODEL_DIR / os.getenv("W2V_MODEL_FILENAME", "word2vec.model")
# Add paths for any other required artifacts (scalers, encoders, etc.)
# SCALER_PATH = MODEL_DIR / "scaler.pkl"

# --- Pydantic Models ---
# Define the input features expected by your model
# Adjust these fields based on your actual model's input
class InputFeatures(BaseModel):
    title: str = Field(..., example="Show HN: My new project using FastAPI and Docker")
    author_karma: int = Field(..., example=1000)
    author_account_age_days: int = Field(..., example=365)
    post_hour: int = Field(..., example=14, ge=0, le=23)
    post_day_of_week: int = Field(..., example=1, ge=0, le=6) # Monday=0, Sunday=6
    # Add other features your model uses (e.g., domain, title length)
    # domain: str = Field(..., example="github.com")

class PredictionResponse(BaseModel):
    predicted_upvotes: float

# --- Global Variables ---
# Initialize app
app = FastAPI(
    title="Hacker News Upvote Predictor API",
    description="Predicts the number of upvotes for a Hacker News post.",
    version="0.1.0"
)

# Load models and other artifacts during startup
model = None
w2v_model = None
# scaler = None

@app.on_event("startup")
async def load_model():
    global model, w2v_model #, scaler
    logger.info(f"Attempting to load model from: {MODEL_PATH}")
    logger.info(f"Attempting to load Word2Vec model from: {W2V_MODEL_PATH}")

    if not MODEL_PATH.exists():
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    if not W2V_MODEL_PATH.exists():
        logger.error(f"Word2Vec model file not found at {W2V_MODEL_PATH}")
        # Decide if Word2Vec is critical - maybe raise RuntimeError
        # raise RuntimeError(f"Word2Vec model file not found at {W2V_MODEL_PATH}")
        pass # Allow startup if W2V is optional or handled differently

    try:
        model = joblib.load(MODEL_PATH)
        logger.info("ML Model loaded successfully.")
        # w2v_model = gensim.models.Word2Vec.load(str(W2V_MODEL_PATH)) # Example for gensim
        # logger.info("Word2Vec Model loaded successfully.")
        # Load scaler if needed
        # if SCALER_PATH.exists():
        #     scaler = joblib.load(SCALER_PATH)
        #     logger.info("Scaler loaded successfully.")

    except Exception as e:
        logger.exception(f"Error loading models: {e}")
        raise RuntimeError(f"Error loading models: {e}")

# --- Helper Functions ---
# Add functions for preprocessing, feature extraction, etc.
def preprocess_input(features: InputFeatures):
    """Preprocesses raw input features into the format expected by the model."""
    logger.info(f"Preprocessing input: {features.dict()}")

    # 1. Extract Word2Vec Embeddings (Placeholder)
    # This needs to be implemented based on your actual W2V usage
    title_embedding = [0.0] * 100 # Replace with actual embedding vector size
    # Example: try:
    #    words = features.title.lower().split()
    #    embeddings = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    #    if embeddings:
    #        title_embedding = pd.DataFrame(embeddings).mean(axis=0).tolist()
    # except Exception as e:
    #    logger.warning(f"Could not generate W2V embedding for title '{features.title}': {e}")
    #    # Handle cases where embedding fails (e.g., return default, raise error)

    # 2. Create DataFrame/Array for model input
    # The order and names must match what the model was trained on
    data = {
        # Add title embedding features (e.g., feature_0, feature_1, ...)
        **{f'title_emb_{i}': emb for i, emb in enumerate(title_embedding)},
        'author_karma': features.author_karma,
        'author_account_age_days': features.author_account_age_days,
        'post_hour': features.post_hour,
        'post_day_of_week': features.post_day_of_week,
        # Add other features from InputFeatures
        # 'domain_feature': encode_domain(features.domain), # Example
        # 'title_length': len(features.title.split())
    }
    df = pd.DataFrame([data])

    # 3. Apply Scaling/Encoding (Placeholder)
    # if scaler:
    #    df = scaler.transform(df) # Ensure columns match scaler's expected input

    logger.info(f"Preprocessing complete. Features shape: {df.shape}")
    return df


# --- API Endpoints ---
@app.get("/", summary="Health Check", description="Check if the API is running.")
async def read_root():
    """Root endpoint for health check."""
    return {"status": "ok", "message": "Hacker News Upvote Predictor API is running."}

@app.post("/predict", response_model=PredictionResponse, summary="Predict Upvotes", description="Predict the number of upvotes for a given Hacker News post.")
async def predict_upvotes(features: InputFeatures):
    """
    Takes post features as input and returns the predicted number of upvotes.
    - **title**: The title of the post.
    - **author_karma**: Karma of the post author.
    - **author_account_age_days**: Age of the author's account in days at the time of posting.
    - **post_hour**: Hour of the day the post was made (0-23).
    - **post_day_of_week**: Day of the week the post was made (0=Monday, 6=Sunday).
    """
    if model is None:
        logger.error("Model is not loaded.")
        raise HTTPException(status_code=503, detail="Model is not loaded. API may be starting up or encountered an error.")

    try:
        # Preprocess the input features
        processed_features = preprocess_input(features)

        # Make prediction
        # Ensure the input format matches the model's expectation (e.g., numpy array, pandas DataFrame)
        prediction = model.predict(processed_features)

        # Extract the prediction value (depends on model output format)
        predicted_value = float(prediction[0])

        logger.info(f"Prediction successful: {predicted_value}")
        return PredictionResponse(predicted_upvotes=predicted_value)

    except Exception as e:
        logger.exception(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# --- Main execution block (for local testing) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server locally for testing...")
    # Run using: python -m src.api.main
    # Note: Model loading might require running from the project root
    # or adjusting MODEL_DIR environment variable/default.
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True) 