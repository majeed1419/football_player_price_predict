from fastapi import FastAPI
import joblib
from pydantic import BaseModel

model = joblib.load('Models/knn.joblib')
scaler = joblib.load('Models/scaler.joblib')

app = FastAPI()
@app.get("/")
def root():
    return "Welcome To Tuwaiq Academy"

# Define a Pydantic model for input data validation
class InputFeatures(BaseModel):
    appearance: int
    minutes_played: int
    current_value: int
    award: int

def preprocessing(input_features: InputFeatures, scaler):
    dict_f = {
        'appearance': input_features.appearance,
        'minutes played': input_features.minutes_played,
        'award': input_features.award,
        'current_value': input_features.current_value,
    }

    # Scale the input features using the provided scaler
    scaled_features = scaler.transform([list(dict_f.values())])
    return scaled_features

@app.post("/predict")
async def predict(input_features: InputFeatures):
    # Call preprocessing function with the scaler
    data = preprocessing(input_features, scaler)
    y_pred = model.predict(data)
    return {"pred": y_pred.tolist()[0]}
