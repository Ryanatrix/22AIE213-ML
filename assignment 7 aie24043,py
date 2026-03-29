import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def A1_load_dataset():
    RAW_EEG_FOLDER = r"C:\Users\rohan\ML_Assignments\EEG_Raw_CSV"
    PARTICIPANTS_FILE = r"C:\Users\rohan\ML_Assignments\participants.tsv"

    participants = pd.read_csv(PARTICIPANTS_FILE, sep="\t")

    features = []

    for file in sorted(os.listdir(RAW_EEG_FOLDER)):
        if not file.endswith(".csv"):
            continue

        subject_id = file.replace("_rawEEG.csv","")
        path = os.path.join(RAW_EEG_FOLDER, file)

        df = pd.read_csv(path, nrows=60000)

        row = {"participant_id": subject_id}

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            row[col] = df[col].mean(skipna=True)

        features.append(row)

    feature_df = pd.DataFrame(features)

    X = feature_df.drop("participant_id", axis=1)
    X = X.select_dtypes(include=[np.number])
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.mean())

    y = participants["MMSE"]

    return X, y


def A2_preprocess(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, le


def A3_split(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def A4_models():
    models = {
        "DecisionTree": (
            DecisionTreeClassifier(),
            {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10]
            }
        ),
        "SVM": (
            SVC(),
            {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"]
            }
        ),
        "RandomForest": (
            RandomForestClassifier(),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10]
            }
        ),
        "AdaBoost": (
            AdaBoostClassifier(),
            {
                "n_estimators": [50, 100],
                "learning_rate": [0.5, 1.0]
            }
        ),
        "NaiveBayes": (
            GaussianNB(),
            {}
        ),
        "MLP": (
            MLPClassifier(max_iter=500),
            {
                "hidden_layer_sizes": [(50,), (100,)],
                "activation": ["relu", "tanh"]
            }
        )
    }

    return models


def A5_tune_and_train(models, X_train, y_train):
    best_models = {}

    for name, (model, params) in models.items():
        if params:
            search = RandomizedSearchCV(
                model,
                params,
                n_iter=5,
                cv=3,
                random_state=42,
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            best_models[name] = search.best_estimator_
        else:
            model.fit(X_train, y_train)
            best_models[name] = model

    return best_models


def A6_evaluate(models, X_train, X_test, y_train, y_test):
    results = []

    for name, model in models.items():
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        row = {
            "Model": name,

            "Train Accuracy": accuracy_score(y_train, y_pred_train),
            "Test Accuracy": accuracy_score(y_test, y_pred_test),

            "Train Precision": precision_score(y_train, y_pred_train, average="weighted", zero_division=0),
            "Test Precision": precision_score(y_test, y_pred_test, average="weighted", zero_division=0),

            "Train Recall": recall_score(y_train, y_pred_train, average="weighted", zero_division=0),
            "Test Recall": recall_score(y_test, y_pred_test, average="weighted", zero_division=0),

            "Train F1": f1_score(y_train, y_pred_train, average="weighted", zero_division=0),
            "Test F1": f1_score(y_test, y_pred_test, average="weighted", zero_division=0),
        }

        results.append(row)

    return pd.DataFrame(results)


def main():
    print("Lab 7 Implementation - Rohan U")

    X, y = A1_load_dataset()
    X, y, le = A2_preprocess(X, y)

    X_train, X_test, y_train, y_test = A3_split(X, y)

    models = A4_models()

    tuned_models = A5_tune_and_train(models, X_train, y_train)

    results = A6_evaluate(tuned_models, X_train, X_test, y_train, y_test)

    print(results)


if __name__ == "__main__":
    main()
