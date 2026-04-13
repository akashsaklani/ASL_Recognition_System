import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle


def train_and_save_model(dataset_path="dataset.csv", model_path="model.pkl"):
    data = pd.read_csv(dataset_path, header=None, on_bad_lines='skip')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    model = RandomForestClassifier()
    model.fit(X, y)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("Model Updated 🔥")


if __name__ == "__main__":
    train_and_save_model()