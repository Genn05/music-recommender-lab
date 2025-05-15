from fastapi import FastAPI
from pydantic import BaseModel
import joblib  # Correct import for joblib

app = FastAPI()

# Load the pre-trained model
try:
    model = joblib.load('music_recommender.joblib')
except FileNotFoundError:
    raise FileNotFoundError("The model file 'music_recommender.joblib' was not found. Ensure the file path is correct.")

# Define input model for API
class UserInput(BaseModel):
    age: int
    gender: int

@app.post('/predict')
def predict(user_input: UserInput):
    try:
        # Predict using the loaded model
        prediction = model.predict([[user_input.age, user_input.gender]])
        return {'genre': prediction[0]}
    except Exception as e:
        return {'error': str(e)}