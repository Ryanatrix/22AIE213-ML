import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

import seaborn as sns
import shap
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

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, le


def A3_split(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def A4_correlation_heatmap(X, feature_names):
    df = pd.DataFrame(X, columns=feature_names)
    corr = df.corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()


def A5_pca(X, variance=0.99):
    pca = PCA(n_components=variance)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def A6_train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model


def A7_feature_selection(X, y, k=5):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new, selector


def A8_lime(pipeline_model, X_train, X_test, feature_names, class_names):
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=[str(c) for c in class_names],
        discretize_continuous=True
    )

    exp = explainer.explain_instance(
        X_test[0],
        pipeline_model.predict_proba
    )

    return exp.as_list()


def A9_shap(model, X_train):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train[:50])
    shap.summary_plot(shap_values, X_train[:50])
    return shap_values


def main():
    X, y = A1_load_dataset()

    feature_names = X.columns.tolist()

    X, y, le = A2_preprocess(X, y)

    X_train, X_test, y_train, y_test = A3_split(X, y)

    A4_correlation_heatmap(X_train, feature_names)

    X_pca_99, pca_99 = A5_pca(X_train, 0.99)
    model_pca_99 = A6_train_model(X_pca_99, y_train)

    X_pca_95, pca_95 = A5_pca(X_train, 0.95)
    model_pca_95 = A6_train_model(X_pca_95, y_train)

    X_fs, selector = A7_feature_selection(X_train, y_train, k=5)
    model_fs = A6_train_model(X_fs, y_train)

    lime_exp = A8_lime(
        model_fs,
        X_train,
        X_test,
        feature_names=[f"f{i}" for i in range(X_train.shape[1])],
        class_names=le.classes_
    )

    shap_values = A9_shap(model_fs, X_train)

    print("PCA 99 shape:", X_pca_99.shape)
    print("PCA 95 shape:", X_pca_95.shape)
    print("Feature Selection shape:", X_fs.shape)
    print("LIME:", lime_exp)


if __name__ == "__main__":
    main()
