from src.utils import deserialize_data
from src.processing import ohe_transform_full
from src.preprocessing import ohe_transform
from pydantic import BaseModel, Field
from fastapi import FastAPI
from contextlib import asynccontextmanager
import pandas as pd
import yaml
import uvicorn
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  

RANDFOR_MODEL_PATH = BASE_DIR / "models" / "randfor_model.pkl"
OHE_HOME_OWNERSHIP_PATH  = BASE_DIR / "models" / "ohe_home_ownership.pkl"
OHE_DEFAULT_ON_FILE_PATH = BASE_DIR / "models" / "ohe_default_on_file.pkl"
OHE_LOAN_GRADE_PATH      = BASE_DIR / "models" / "ohe_loan_grade.pkl"
OHE_LOAN_INTENT_PATH     = BASE_DIR / "models" / "ohe_loan_intent.pkl"


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

@app.post('/home')
def home():
    return {"message" : "Hello World"}

@app.post('/predict')
def predict(item: Item):
    ohe_transform_ownership = deserialize_data(OHE_HOME_OWNERSHIP_PATH)
    ohe_loan_intent = deserialize_data(OHE_LOAN_INTENT_PATH)
    ohe_loan_grade = deserialize_data(OHE_LOAN_GRADE_PATH)
    ohe_default_on_file = deserialize_data(OHE_DEFAULT_ON_FILE_PATH)

    df = pd.DataFrame([item.model_dump()])
    df = ohe_transform(df, 'person_home_ownership', 'home_ownership', ohe_transform_ownership)
    df = ohe_transform(df, 'loan_intent', 'loan_intent', ohe_loan_intent)
    df = ohe_transform(df, 'loan_grade', 'loan_grade', ohe_loan_grade)
    df = ohe_transform(df, 'cb_person_default_on_file', 'default_onfile', ohe_default_on_file)
    # df = ohe_transform_full(df)

    THRESHOLD = 0.3131 

    proba = ml_model["model"].predict_proba(df)  
    prob_default = proba[0][1]                  

    prediction = int(prob_default >= THRESHOLD)  

    return {
        "prediction": prediction,
        "probability_default": round(float(prob_default), 4),
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)