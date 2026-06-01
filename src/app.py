import pickle
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
import sys

app = FastAPI(title="NVIDIA Stock Price Prediction Service")

# 1. Keep the identical Neural Network blueprint
class StockRegressor(nn.Module):
    def __init__(self, input_dim):
        super(StockRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Explicitly map the class to the runtime entrypoint so Pickle doesn't get lost
sys.modules['__main__'].StockRegressor = StockRegressor

class PredictionInput(BaseModel):
    lag_1: float
    daily_return: float
    ma_5: float

@app.get("/")
def home():
    return {"status": "healthy", "model": "StockRegressor-v1"}
@app.post("/predict")
def predict_next_day(data: PredictionInput):
    # 1. Custom Unpickler to catch and fix module namespace paths on the fly
    class ContainerUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # If pickle is looking for StockRegressor inside old modules, redirect it here
            if name == 'StockRegressor':
                return StockRegressor
            # Force storage to redirect to CPU arrays
            if module == 'torch.storage' and name == '_load_from_bytes':
                import io
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            return super().find_class(module, name)

    # 2. Open and read the artifact cleanly using our custom routing rules
    with open("data/model.pkl", "rb") as f:
        loaded_model = ContainerUnpickler(f).load()

    # 3. Check if we extracted raw weights or a full model class object
    if isinstance(loaded_model, dict):
        model = StockRegressor(input_dim=3)
        model.load_state_dict(loaded_model)
    elif hasattr(loaded_model, 'state_dict'):
        # Extract weights from the pickled object and load them into our fresh local class setup
        model = StockRegressor(input_dim=3)
        model.load_state_dict(loaded_model.state_dict())
    else:
        model = loaded_model

    model.eval()
    
    # 4. Process incoming inference numbers
    input_features = np.array([[data.lag_1, data.daily_return, data.ma_5]])
    input_tensor = torch.tensor(input_features, dtype=torch.float32)
    
    with torch.no_grad():
        prediction = model(input_tensor)
        
    return {"predicted_next_day_close": float(prediction.item())}

