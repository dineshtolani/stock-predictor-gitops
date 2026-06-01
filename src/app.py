import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Stock Predictor API", version="1.0")

# Define the input payload structure matching the model features
class PredictionInput(BaseModel):
    lag_1: float
    daily_return: float
    ma_5: float

# Define model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "model.pkl")

# Load model weights on startup
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    print(f"Warning: Model weights not found at {MODEL_PATH}")

@app.get("/")
def read_root():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model weight matrix not loaded initialization failure.")
    
    try:
        # Pass features directly to the model structure
        # (Assuming a basic PyTorch regression setup or standard scikit-style model)
        inputs = [data.lag_1, data.daily_return, data.ma_5]
        
        # If it's a PyTorch model, we execute inference. If it's standard scikit/joblib framework:
        if hasattr(model, 'predict'):
            prediction = model.predict([inputs])[0]
        else:
            import torch
            with torch.no_grad():
                tensor_input = torch.tensor([inputs], dtype=torch.float32)
                prediction = model(tensor_input).item()
                
        return {"predicted_next_day_close": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
