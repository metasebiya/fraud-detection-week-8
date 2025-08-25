from fastapi import FastAPI
from pydantic import BaseModel
import joblib  # or your model loader

app = FastAPI()
model = joblib.load("models/fraud_model.pkl")

class InputData(BaseModel):
    feature1: float
    feature2: str
    # Add all required features

@app.post("/predict")
def predict(data: InputData):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict_proba(input_df)[0][1]
    return {"fraud_score": prediction}
