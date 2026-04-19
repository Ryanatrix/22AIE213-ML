import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from lime.lime_tabular import LimeTabularExplainer


def A1_load_dataset():
    RAW_EEG_FOLDER = r"C:\Users\rohan\ML_Assignments\EEG_Raw_CSV"
    PARTICIPANTS_FILE = r"C:\Users\rohan\ML_Assignments\participants.tsv"

    participants = pd.read_csv(PARTICIPANTS_FILE, sep="\t")

    features = []

    for file in sorted(os.listdir(RAW_EEG_FOLDER)):
        if file.endswith(".csv"):
            subject_id = file.replace("_rawEEG.csv","")
            df = pd.read_csv(os.path.join(RAW_EEG_FOLDER, file), nrows=60000)

            row = {"participant_id": subject_id}

            for col in df.select_dtypes(include=[np.number]).columns:
                row[col] = df[col].mean()

            features.append(row)

    feature_df = pd.DataFrame(features)

    X = feature_df.drop("participant_id", axis=1)
    X = X.fillna(X.mean())

    y = participants["MMSE"]

    return X, y


def A2_preprocess(X, y):
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X.values, y, le


def A3_split(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def A4_stacking_model():
    base_models = [
        ("dt", DecisionTreeClassifier(max_depth=5)),
        ("svm", SVC(probability=True)),
        ("rf", RandomForestClassifier(n_estimators=100))
    ]

    meta_model = LogisticRegression()

    model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model
    )

    return model


def A5_pipeline(model):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])

    return pipeline


def A6_train(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


def A7_predict(pipeline, X_test):
    return pipeline.predict(X_test)


def A8_lime_explainer(pipeline, X_train, X_test, feature_names, class_names):
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=[str(c) for c in class_names],
        discretize_continuous=True
    )

    exp = explainer.explain_instance(
        X_test[0],
        pipeline.predict_proba
    )

    return exp.as_list()


def main():
    X, y = A1_load_dataset()

    X, y, le = A2_preprocess(X, y)

    X_train, X_test, y_train, y_test = A3_split(X, y)

    model = A4_stacking_model()

    pipeline = A5_pipeline(model)

    pipeline = A6_train(pipeline, X_train, y_train)

    preds = A7_predict(pipeline, X_test)

    explanation = A8_lime_explainer(
        pipeline,
        X_train,
        X_test,
        feature_names=[f"f{i}" for i in range(X.shape[1])],
        class_names=le.classes_
    )

    print("Predictions:", preds[:5])
    print("LIME Explanation:", explanation)


if __name__ == "__main__":
    main()
