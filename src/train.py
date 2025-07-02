import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    # Load data
    df = pd.read_csv("data/spam.csv")
    df.drop(columns=['Email No.'], inplace=True)

    X = df.drop(columns=['Prediction'])
    y = df['Prediction']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save model
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/spam_naive_bayes_model.pkl")
    print("Model saved!")

if __name__ == "__main__":
    main()
