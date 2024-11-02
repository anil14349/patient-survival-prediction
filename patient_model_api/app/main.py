import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import gradio as gr
import joblib
import numpy as np

xgb_clf = joblib.load('./patient_model/trained_models/xgboost-model.pkl')

def predict_death_event(age, anaemia, high_blood_pressure, creatinine_phosphokinase, diabetes, ejection_fraction, platelets, sex, serum_creatinine, serum_sodium, smoking, time):
    '''Function to predict survival of patients with heart failure'''
    test_data = np.array([[age, anaemia, high_blood_pressure, creatinine_phosphokinase, diabetes, ejection_fraction, platelets, sex, serum_creatinine, serum_sodium, smoking, time]])
    prediction = xgb_clf.predict(test_data)
    if prediction == 0:
        return "Patient is not dead"
    else:
        return "Patient is dead"
    
    

input_components = [
    gr.Slider(1, 100, value=4, label="Age",),
    gr.Checkbox(label="Anaemia"),
    gr.Checkbox(label="High blood pressure"),
    gr.Slider(0,10000,label="Creatinine phosphokinase"),
    gr.Checkbox(label="Diabetes"),
    gr.Slider(0,200,label="Ejection fraction"),
    gr.Slider(0,40000000,label="Platelets"),
    gr.Checkbox(label="Sex"),
    gr.Slider(0,10000,label="Serum creatinine"),
    gr.Slider(0,10000,label="Serum sodium"),
    gr.Checkbox(label="Smoking"),
    gr.Slider(0,10000,label="Time")
]

output_component = gr.Label(label="Survival Prediction")





if __name__ == "__main__":
    # Gradio interface to generate UI link
    title = "Patient Survival Prediction"
    description = "Predict survival of patient with heart failure, given their clinical record"

    iface = gr.Interface(fn = predict_death_event,
                            inputs = input_components,
                            outputs = output_component,
                            title = title,
                            description = description,
                            allow_flagging='never')

    iface.launch(share = True, server_name="0.0.0.0", server_port = 8001)   # Ref: https://www.gradio.app/docs/interface
