import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    # 1. Load Dataset
    data_path = os.path.join(os.path.dirname(__file__), 'placement.csv')
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # 2. Preprocess
    # Drop first unnamed index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    print("\nDataset overview:")
    print(df.head())
    print(f"\nShape: {df.shape}")

    # 3. Extract inputs (X) and target (y)
    X = df[['cgpa', 'iq']]
    y = df['placement']

    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # 5. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Model Training (Logistic Regression)
    clf = LogisticRegression()
    clf.fit(X_train_scaled, y_train)
    print("\nModel training completed.")

    # 7. Evaluation
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on test set: {accuracy * 100:.2f}%")

    # 8. Save Model and Scaler
    model_file = os.path.join(os.path.dirname(__file__), 'model.pkl')
    scaler_file = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

    with open(model_file, 'wb') as f:
        pickle.dump(clf, f)
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\nSaved model to '{model_file}' and scaler to '{scaler_file}'")

    # 9. Test a sample prediction
    sample_data = pd.DataFrame([[8.2, 120.0]], columns=['cgpa', 'iq'])
    sample_scaled = scaler.transform(sample_data)
    prediction = clf.predict(sample_scaled)[0]
    prob = clf.predict_proba(sample_scaled)[0][1]

    status = "Placed" if prediction == 1 else "Not Placed"
    print(f"\n--- Sample Prediction ---")
    print(f"CGPA: {sample_data.iloc[0]['cgpa']}, IQ: {sample_data.iloc[0]['iq']}")
    print(f"Result: {status} (Placement Probability: {prob * 100:.2f}%)")

if __name__ == '__main__':
    main()
