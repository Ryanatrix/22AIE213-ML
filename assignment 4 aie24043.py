import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report


# 
# A1 
def A1_confusion_and_metrics():
    #  Similar train and test performance indicates a regular-fit kNN model.

    df = pd.read_excel("participants_with_eeg_features.xlsx")

    X_data = df.drop(columns=["participant_id", "Group"])
    y_labels = LabelEncoder().fit_transform(df["Group"])

    for col in X_data.select_dtypes(include=["object"]).columns:
        X_data[col] = LabelEncoder().fit_transform(X_data[col])

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    print("A1 TRAIN CONFUSION MATRIX")
    print(confusion_matrix(y_train, knn_model.predict(X_train)))
    print(classification_report(y_train, knn_model.predict(X_train)))

    print("A1 TEST CONFUSION MATRIX")
    print(confusion_matrix(y_test, knn_model.predict(X_test)))
    print(classification_report(y_test, knn_model.predict(X_test)))


# 
# A2 
def A2_regression_metrics():
    #  Lower RMSE and higher R2 indicate better regression performance.

    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")

    df["Target_Label"] = np.where(df["Payment (Rs)"] > 200, 1, 0)

    X_vals = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y_vals = df["Target_Label"].values

    X_vals = np.column_stack((np.ones(len(X_vals)), X_vals))
    weights = np.linalg.pinv(X_vals) @ y_vals
    predictions = X_vals @ weights

    mse = np.mean((y_vals - predictions) ** 2)
    rmse = np.sqrt(mse)

    valid_idx = y_vals != 0
    mape = np.mean(np.abs((y_vals[valid_idx] - predictions[valid_idx]) / y_vals[valid_idx])) * 100

    ss_res = np.sum((y_vals - predictions) ** 2)
    ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print("A2 MSE :", mse)
    print("A2 RMSE:", rmse)
    print("A2 MAPE:", mape)
    print("A2 R2  :", r2)


# 
# A3 
def A3_generate_training_data():
    #  Data points form two separable clusters based on linear boundary.

    np.random.seed(7)

    x_train = np.random.randint(1, 11, 20)
    y_train = np.random.randint(1, 11, 20)

    class_tags = []
    for i in range(len(x_train)):
        class_tags.append(0 if x_train[i] + y_train[i] <= 11 else 1)

    plt.scatter(x_train, y_train, c=class_tags, edgecolor="black", s=80)
    plt.xlabel("X Feature")
    plt.ylabel("Y Feature")
    plt.title("A3 Training Data Distribution")
    plt.show()

    return np.column_stack((x_train, y_train)), np.array(class_tags)


# 
# A4 
def A4_test_grid_knn(train_points, train_labels, k):
    #  kNN creates nonlinear decision boundaries in feature space.

    x_range = np.arange(0, 10, 0.1)
    y_range = np.arange(0, 10, 0.1)
    gx, gy = np.meshgrid(x_range, y_range)

    test_points = np.c_[gx.ravel(), gy.ravel()]

    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(train_points, train_labels)

    predictions = knn_model.predict(test_points)

    plt.scatter(test_points[predictions == 0, 0],
                test_points[predictions == 0, 1],
                color="blue", s=5)

    plt.scatter(test_points[predictions == 1, 0],
                test_points[predictions == 1, 1],
                color="red", s=5)

    plt.scatter(train_points[:, 0], train_points[:, 1],
                c=train_labels, edgecolor="black", s=80)

    plt.title(f"A4 kNN Decision Regions (k={k})")
    plt.xlabel("X Feature")
    plt.ylabel("Y Feature")
    plt.show()


# 
# A5 
def A5_vary_k_effect(train_points, train_labels):
    #  Increasing k smoothens boundaries and reduces overfitting.

    for k in range(1, 8):
        A4_test_grid_knn(train_points, train_labels, k)


# 
# A6 
def A6_project_data_knn():
    #  Project features show partial overlap but remain classifiable.

    df = pd.read_excel("participants_with_eeg_features.xlsx")

    X_proj = df[["Relative_Alpha", "Alpha_Theta_Ratio"]]
    y_proj = LabelEncoder().fit_transform(df["Group"])

    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_proj, y_proj)

    x_min, x_max = X_proj.iloc[:, 0].min(), X_proj.iloc[:, 0].max()
    y_min, y_max = X_proj.iloc[:, 1].min(), X_proj.iloc[:, 1].max()

    gx, gy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[gx.ravel(), gy.ravel()]
    pred = knn_model.predict(grid)

    plt.scatter(grid[pred == 0, 0], grid[pred == 0, 1],
                color="blue", s=5, alpha=0.3)
    plt.scatter(grid[pred == 1, 0], grid[pred == 1, 1],
                color="red", s=5, alpha=0.3)

    plt.scatter(X_proj.iloc[:, 0], X_proj.iloc[:, 1],
                c=y_proj, edgecolor="black", s=80)

    plt.xlabel("Relative Alpha")
    plt.ylabel("Alpha / Theta Ratio")
    plt.title("A6 Project Data kNN")
    plt.show()


# 
# A7 
def A7_hyperparameter_tuning():
    #  Optimal k improves generalization by balancing bias and variance.

    df = pd.read_excel("participants_with_eeg_features.xlsx")

    X = df.drop(columns=["participant_id", "Group"])
    y = LabelEncoder().fit_transform(df["Group"])

    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    param_grid = {"n_neighbors": [1, 3, 5, 7, 9, 11]}

    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring="accuracy"
    )

    grid.fit(X_scaled, y)

    print("A7 BEST k:", grid.best_params_["n_neighbors"])
    print("A7 BEST CV ACCURACY:", grid.best_score_)


# 
# MAIN 
def main():
    #this code was written and developed by Rohan U(bl.sc.u4aie24043)
    A1_confusion_and_metrics()
    A2_regression_metrics()

    train_pts, train_lbls = A3_generate_training_data()
    A4_test_grid_knn(train_pts, train_lbls, k=3)
    A5_vary_k_effect(train_pts, train_lbls)

    A6_project_data_knn()
    A7_hyperparameter_tuning()


main()
