# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 18:50:57 2022

@author: siddhardhan
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle

# Load the trained model
model = pickle.load(open('loan_status_prediction_rs.sav', 'rb'))

# Initialize FastAPI app
app = FastAPI()

# Define the request body using Pydantic
class PredictionRequest(BaseModel):
    Gender: int
    Married: int
    Dependents: int
    Education: int
    Self_Employed: int
    ApplicantIncome: int
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: int

@app.post('/predict')
def predict(request: PredictionRequest):
    # Extract features from the request data
    features = [
        request.Gender,
        request.Married,
        request.Dependents,
        request.Education,
        request.Self_Employed,
        request.ApplicantIncome,
        request.CoapplicantIncome,
        request.LoanAmount,
        request.Loan_Amount_Term,
        request.Credit_History,
        request.Property_Area
    ]
    
    # Make prediction
    try:
        prediction = model.predict([features])
        return {'prediction': int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

    
    



