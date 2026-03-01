import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

RAW_EEG_FOLDER = r"C:\Users\rohan\Downloads\ML_Assignments\EEG_Raw_CSV"
PARTICIPANTS_FILE = r"C:\Users\rohan\Downloads\ML_Assignments\participants.tsv"


def build_feature_dataset(raw_folder, participants_file, target_column="MMSE"):
    participants = pd.read_csv(participants_file, sep="\t")
    participants = participants.sort_values("participant_id")

    feature_rows = []

    for file in sorted(os.listdir(raw_folder)):
        if not file.endswith(".csv"):
            continue

        subject_id = file.replace("_rawEEG.csv", "")
        file_path = os.path.join(raw_folder, file)

        df = pd.read_csv(file_path, nrows=60000)
        means = df.mean()

        subject_features = {"participant_id": subject_id}
        for col in df.columns:
            subject_features[col] = means[col]

        feature_rows.append(subject_features)

    feature_df = pd.DataFrame(feature_rows)

    dataset = feature_df.merge(
        participants[["participant_id", target_column]],
        on="participant_id"
    )

    dataset.rename(columns={target_column: "target"}, inplace=True)

    X = dataset.drop(columns=["participant_id", "target"]).values
    y = dataset["target"].values

    return X, y


def calculate_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2


def a1_linear_regression_single_feature(X, y):
    X_single = X[:, 0].reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_single, y, test_size=0.2, random_state=42
    )

    model = LinearRegression().fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return y_train, y_test, y_train_pred, y_test_pred


def a2_evaluate_single_feature_model(y_train, y_test, y_train_pred, y_test_pred):
    train_metrics = calculate_regression_metrics(y_train, y_train_pred)
    test_metrics = calculate_regression_metrics(y_test, y_test_pred)
    return train_metrics, test_metrics


def a3_linear_regression_all_features(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression().fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = calculate_regression_metrics(y_train, y_train_pred)
    test_metrics = calculate_regression_metrics(y_test, y_test_pred)

    return train_metrics, test_metrics


def a4_kmeans_clustering_k2(X):
    kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(X)
    return kmeans.labels_, kmeans.cluster_centers_


def a5_clustering_evaluation_scores(X, labels):
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    return sil, ch, db


def a6_kmeans_multiple_k_evaluation(X, k_range=range(2, 10)):
    sil_scores = []
    ch_scores = []
    db_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
        labels = km.labels_

        sil_scores.append(silhouette_score(X, labels))
        ch_scores.append(calinski_harabasz_score(X, labels))
        db_scores.append(davies_bouldin_score(X, labels))

    return list(k_range), sil_scores, ch_scores, db_scores


def a7_elbow_method(X, k_range=range(2, 20)):
    distortions = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
        distortions.append(km.inertia_)

    return list(k_range), distortions


def main():
    print("this code was written and developed by ROHAN U (bl.sc.u4aie24043) ")
    X, y = build_feature_dataset(RAW_EEG_FOLDER, PARTICIPANTS_FILE)

    print("A1: Linear Regression (Single Feature)")
    y_train, y_test, y_train_pred, y_test_pred = a1_linear_regression_single_feature(X, y)

    print("A2: Evaluation Metrics (Single Feature)")
    train_metrics, test_metrics = a2_evaluate_single_feature_model(
        y_train, y_test, y_train_pred, y_test_pred
    )
    print("Train Metrics:", train_metrics)
    print("Test Metrics:", test_metrics)

    print("A3: Linear Regression (All Features)")
    train_metrics_all, test_metrics_all = a3_linear_regression_all_features(X, y)
    print("Train Metrics:", train_metrics_all)
    print("Test Metrics:", test_metrics_all)

    print("A4: K-Means Clustering (k=2)")
    labels, centers = a4_kmeans_clustering_k2(X)
    print("Cluster Centers:\n", centers)

    print("A5: Clustering Evaluation Scores")
    sil, ch, db = a5_clustering_evaluation_scores(X, labels)
    print("Silhouette Score:", sil)
    print("Calinski-Harabasz Score:", ch)
    print("Davies-Bouldin Index:", db)

    print("A6: Evaluation for Multiple k Values")
    k_vals, sil_scores, ch_scores, db_scores = a6_kmeans_multiple_k_evaluation(X)

    plt.figure()
    plt.plot(k_vals, sil_scores, label="Silhouette")
    plt.plot(k_vals, ch_scores, label="CH Score")
    plt.plot(k_vals, db_scores, label="DB Index")
    plt.xlabel("k")
    plt.title("Clustering Evaluation Scores")
    plt.legend()
    plt.show()

    print("A7: Elbow Plot")
    k_vals_elbow, distortions = a7_elbow_method(X)

    plt.figure()
    plt.plot(k_vals_elbow, distortions)
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow Plot")
    plt.show()


if __name__ == "__main__":
    main()