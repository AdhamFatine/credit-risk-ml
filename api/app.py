from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

import os, joblib

model_path = os.path.join(os.path.dirname(__file__), "../model/credit_model.pkl")
model = joblib.load(model_path)

# create FastAPI app
app = FastAPI()

# define input schema
class ClientData(BaseModel):
    status_account: int
    month_duration: int
    credit_history: int
    purpose: int
    credit_amount: int
    status_savings: int
    years_employment: int
    payment_to_income_ratio: int
    status_and_sex: int
    secondary_obligor: int
    residence_since: int
    collateral: int
    age: int
    other_installment_plans: int
    housing: int
    n_credits: int
    job: int
    n_guarantors: int
    telephone: int
    is_foreign_worker: int

@app.post("/predict")
def predict_credit(data: ClientData):
    features = np.array([[data.status_account, data.month_duration, data.credit_history,
                          data.purpose, data.credit_amount, data.status_savings,
                          data.years_employment, data.payment_to_income_ratio,
                          data.status_and_sex, data.secondary_obligor,
                          data.residence_since, data.collateral, data.age,
                          data.other_installment_plans, data.housing, data.n_credits,
                          data.job, data.n_guarantors, data.telephone, data.is_foreign_worker]])
    prediction = model.predict(features)[0]
    result = "Crédit accepté" if prediction == 1 else "Crédit refusé"
    return {"prediction": result}