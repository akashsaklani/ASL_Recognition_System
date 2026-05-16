import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_and_save_model(dataset_path="dataset.csv", model_path="model.pkl"):

    # Load dataset
    data = pd.read_csv(dataset_path, header=None, on_bad_lines='skip')

    # Features and labels
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Random Forest Model
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("\nModel Saved Successfully 🔥")


if __name__ == "__main__":
    train_and_save_model()