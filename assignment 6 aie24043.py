import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder


def A1_entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    ent = 0
    for p in probs:
        ent -= p * np.log2(p)
    return ent


def A1_equal_width_binning(series, bins=4):
    series = pd.to_numeric(series, errors='coerce')
    series = series.dropna()

    if len(series) == 0:
        return pd.Series([], dtype='int64')

    min_val = series.min()
    max_val = series.max()

    if min_val == max_val:
        return pd.Series([0]*len(series), index=series.index)

    width = (max_val - min_val) / bins
    edges = [min_val + i*width for i in range(bins+1)]

    return pd.cut(series, bins=edges, labels=False, include_lowest=True)


def A2_gini_index(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs**2)


def A3_information_gain(X_col, y):
    parent_entropy = A1_entropy(y)

    mask = ~pd.isna(X_col)
    X_col = X_col[mask]
    y = y[mask]

    values, counts = np.unique(X_col, return_counts=True)

    weighted_entropy = 0
    for v, c in zip(values, counts):
        subset_y = y[X_col == v]
        weighted_entropy += (c/len(X_col)) * A1_entropy(subset_y)

    return parent_entropy - weighted_entropy


def A3_root_feature(X, y):
    gains = {}

    for col in X.columns:
        binned = A1_equal_width_binning(X[col])
        if len(binned) > 0:
            gains[col] = A3_information_gain(binned, y)
        else:
            gains[col] = 0

    root = max(gains, key=gains.get)
    return root, gains


def A5_build_decision_tree(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X, y_encoded)

    return dt, le, y_encoded


def A6_visualize_tree(dt, X, le):
    plt.figure(figsize=(15,10))
    plot_tree(
        dt,
        feature_names=X.columns.tolist(),
        class_names=[str(c) for c in le.classes_],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title("Decision Tree")
    plt.tight_layout()
    plt.show()


def A7_decision_boundary(X, y_encoded):
    if X.shape[1] < 2:
        return

    features = X.iloc[:, :2].values

    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(features, y_encoded)

    x_min, x_max = features[:,0].min()-1, features[:,0].max()+1
    y_min, y_max = features[:,1].min()-1, features[:,1].max()+1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10,8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap="RdYlBu")

    scatter = plt.scatter(
        features[:,0],
        features[:,1],
        c=y_encoded,
        cmap="RdYlBu",
        edgecolors="black"
    )

    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.title("Decision Boundary (Decision Tree)")
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()


def load_dataset():
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


def main():
    print("this code was written and devloped by Rohan U(BL.SC.U4AIE24043)")
    X, y = load_dataset()

    print("Entropy:", A1_entropy(y))
    print("Gini Index:", A2_gini_index(y))

    root, gains = A3_root_feature(X, y)

    print("Root Feature:", root)

    for f,g in gains.items():
        print(f, g)

    dt, le, y_encoded = A5_build_decision_tree(X, y)

    A6_visualize_tree(dt, X, le)

    A7_decision_boundary(X, y_encoded)


if __name__ == "__main__":
    main()