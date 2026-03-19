from src.utils import deserialize_data
from src.processing import ohe_transform_full
from pydantic import BaseModel, Field
from fastapi import FastAPI
from contextlib import asynccontextmanager
import pandas as pd
import yaml
import uvicorn
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  

RANDFOR_MODEL_PATH = BASE_DIR / "models" / "randfor_model.pkl"

def load_model():
    return deserialize_data(RANDFOR_MODEL_PATH)

ml_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model["model"] = load_model()
    yield
    ml_model.clear()

class Item (BaseModel):
    person_age : int
    person_income : int
    person_home_ownership : str
    person_emp_length : float
    loan_intent : str
    loan_grade : str
    loan_amnt : int
    loan_int_rate : float
    loan_percent_income: float
    cb_person_default_on_file : str
    cb_person_cred_hist_length : int

app = FastAPI(lifespan= lifespan)

@app.get("/")
def root():
    return {"message": "ML Prediction API is running", "docs": "/docs"}

@app.get("/best_model")
def check_best_model():
    model = load_model()
    return {f"This is best model : {model.best_estimator_.__class__.__name__}"}

@app.post('/home')
def home():
    return {"message" : "Hello World"}

@app.post('/predict')
def predict(item: Item):
    df = pd.DataFrame([item.model_dump()])
    df = ohe_transform_full(df)
    prediction = ml_model["model"].predict(df)
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)