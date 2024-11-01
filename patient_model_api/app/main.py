import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import gradio as gr
import joblib
import numpy as np
from fastapi import FastAPI

# FastAPI object
app = FastAPI()
xgb_clf = joblib.load(str(root.parent / 'patient_model/trained_models/xgboost-model.pkl'))

def predict_death_event(age, anaemia, high_blood_pressure, creatinine_phosphokinase, diabetes, ejection_fraction, platelets, sex, serum_creatinine, serum_sodium, smoking, time):
    '''Function to predict survival of patients with heart failure'''
    test_data = np.array([[age, anaemia, high_blood_pressure, creatinine_phosphokinase, diabetes, ejection_fraction, platelets, sex, serum_creatinine, serum_sodium, smoking, time]])
    prediction = xgb_clf.predict(test_data)
    if prediction == 0:
        return "Patient is not dead"
    else:
        return "Patient is dead"
    
    

input_components = [
    gr.Number(label="Age"),
    gr.Number(label="Creatinine Phosphokinase"),
    gr.Number(label="Ejection Fraction"),
    gr.Number(label="Platelets"),
    gr.Number(label="Serum Creatinine"),
    gr.Number(label="Serum Sodium"),
    gr.Checkbox(label="High Blood Pressure"),
    gr.Checkbox(label="Diabetes"),
    gr.Checkbox(label="Smoking"),
]

output_component = gr.Label(label="Survival Prediction")


# Gradio interface to generate UI link
title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gr.Interface(fn = predict_death_event,
                         inputs = input_components,
                         outputs = output_component,
                         title = title,
                         description = description,
                         allow_flagging='never')

iface.launch(share = True)  # server_name="0.0.0.0", server_port = 8001   # Ref: https://www.gradio.app/docs/interface


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 