import os
import pickle
import pandas as pd

def predict_placement(cgpa: float, iq: float):
    model_file = os.path.join(os.path.dirname(__file__), 'model.pkl')
    scaler_file = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        print("Model or Scaler not found! Run `python train.py` first.")
        return

    with open(model_file, 'rb') as f:
        clf = pickle.load(f)
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)

    # Prepare input
    data = pd.DataFrame([[cgpa, iq]], columns=['cgpa', 'iq'])
    data_scaled = scaler.transform(data)

    prediction = clf.predict(data_scaled)[0]
    prob = clf.predict_proba(data_scaled)[0][1]

    status = "Placed" if prediction == 1 else "Not Placed"
    print(f"CGPA: {cgpa}, IQ: {iq} -> {status} (Probability: {prob * 100:.2f}%)")
    return prediction, prob

if __name__ == '__main__':
    print("--- Placement Predictions ---")
    predict_placement(cgpa=8.5, iq=115.0)
    predict_placement(cgpa=5.2, iq=95.0)
    predict_placement(cgpa=6.8, iq=123.0)
