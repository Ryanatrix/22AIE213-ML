import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import minkowski as scipy_minkowski
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder


#  A1 
def a1_vector_operations(A, B): #Implemented dot product and Euclidean norm manually and validated results using NumPy functions.
    own_dot = sum(A[i] * B[i] for i in range(len(A)))
    own_norm_A = math.sqrt(sum(x ** 2 for x in A))
    own_norm_B = math.sqrt(sum(x ** 2 for x in B))

    np_dot = np.dot(A, B)
    np_norm_A = np.linalg.norm(A)
    np_norm_B = np.linalg.norm(B)

    return own_dot, own_norm_A, own_norm_B, np_dot, np_norm_A, np_norm_B


#  A2 
def a2_intra_inter_class(df, features): #Computed class centroids and spreads using mean and standard deviation to analyze intra- and inter-class distances.
    classes = df["Mental_State"].unique()[:2]

    X1 = df[df["Mental_State"] == classes[0]][features].values
    X2 = df[df["Mental_State"] == classes[1]][features].values

    centroid1 = np.mean(X1, axis=0)
    centroid2 = np.mean(X2, axis=0)

    spread1 = np.std(X1, axis=0)
    spread2 = np.std(X2, axis=0)

    interclass_distance = np.linalg.norm(centroid1 - centroid2)

    return centroid1, centroid2, spread1, spread2, interclass_distance


#  A3 
def a3_density_pattern(df, feature): # numpy.histogram to study feature density and calculated mean and variance.
    values = df[feature].dropna().values

    hist_values, bin_edges = np.histogram(values, bins=10)
    mean_val = np.mean(values)
    var_val = np.var(values)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.bar(bin_centers, hist_values, width=(bin_edges[1] - bin_edges[0]))
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {feature}")
    plt.show()

    return mean_val, var_val


#  A4 
def a4_minkowski_custom(v1, v2): #Implemented Minkowski distance for varying p values to observe distance behavior.
    distances = []
    for p in range(1, 11):
        dist = sum(abs(v1[i] - v2[i]) ** p for i in range(len(v1))) ** (1 / p)
        distances.append(dist)
    return distances


#  A5 
def a5_minkowski_comparison(v1, v2): #Compared custom Minkowski distance with SciPy’s built-in implementation for validation.
    own = a4_minkowski_custom(v1, v2)
    scipy_vals = [scipy_minkowski(v1, v2, p) for p in range(1, 11)]
    return own, scipy_vals


#  A6 
def a6_train_test_split(df, features): #Split dataset into training and testing sets using train_test_split.
    X = df[features].values
    y = df["Mental_State"].values
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


#  A7 
def a7_knn_training(X_train, y_train): #Trained a kNN classifier with k=3 using the training dataset.
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    return model


#  A8 
def a8_knn_accuracy(model, X_test, y_test): #Evaluated kNN classifier accuracy using the built-in score() method.
    return model.score(X_test, y_test)


#  A9 
def a9_knn_predictions(model, X_test): #Used predict() to study classifier prediction behavior on test data.
    return model.predict(X_test)


#  A10 
def a10_custom_knn(X_train, y_train, X_test, k=3): #Implemented kNN classification manually and compared it with sklearn’s kNN.
    predictions = []
    for xt in X_test:
        distances = []
        for i in range(len(X_train)):
            d = np.linalg.norm(xt - X_train[i])
            distances.append((d, y_train[i]))
        distances.sort(key=lambda x: x[0])
        labels = [l for _, l in distances[:k]]
        predictions.append(max(set(labels), key=labels.count))
    return np.array(predictions)


#  A11 
def a11_accuracy_vs_k(X_train, y_train, X_test, y_test): #Analyzed the effect of k on classification accuracy and plotted accuracy vs k.
    acc = []
    for k in range(1, 12):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        acc.append(model.score(X_test, y_test))
    return acc


#  A12 
def a12_confusion_and_metrics(y_true, y_pred): #Evaluated confusion matrix and derived precision, recall, and F1-score.
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    return cm, precision, recall, f1


#  A13 
def a13_metrics_from_confusion(cm): #Computed accuracy, precision, recall, and F1-score manually from confusion matrix.
    accuracy = np.trace(cm) / np.sum(cm)
    precision = np.mean(np.diag(cm) / np.sum(cm, axis=0))
    recall = np.mean(np.diag(cm) / np.sum(cm, axis=1))
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1


#  A14 
def a14_matrix_inversion_classifier(X_train, y_train, X_test): #Compared kNN performance with a matrix inversion–based classifier.
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    Y = np.eye(len(np.unique(y_enc)))[y_enc]
    W = np.linalg.pinv(X_train) @ Y

    scores = X_test @ W
    predictions = le.inverse_transform(np.argmax(scores, axis=1))
    return predictions


#  MAIN 
def main():
    df = pd.read_excel(r"C:\Users\rohan\Downloads\ml assignments\dataset2.xlsx")

    features = [c for c in df.columns if c.startswith("EEG")]

    A = np.array([1, 2, 3])
    B = np.array([5, 6, 7])
    print("A1:", a1_vector_operations(A, B))

    print("A2:", a2_intra_inter_class(df, features))

    print("A3:", a3_density_pattern(df, "EEG2"))

    v1 = df.loc[0, features].values
    v2 = df.loc[1, features].values
    print("A4:", a4_minkowski_custom(v1, v2))
    print("A5:", a5_minkowski_comparison(v1, v2))

    X_train, X_test, y_train, y_test = a6_train_test_split(df, features)

    knn = a7_knn_training(X_train, y_train)
    print("A8 Accuracy:", a8_knn_accuracy(knn, X_test, y_test))

    y_test_pred = a9_knn_predictions(knn, X_test)
    print("A9 Predictions:", y_test_pred)

    custom_pred = a10_custom_knn(X_train, y_train, X_test)
    print("A10 Accuracy:", np.mean(custom_pred == y_test))

    acc_values = a11_accuracy_vs_k(X_train, y_train, X_test, y_test)
    print("A11 Accuracies:", acc_values)

    plt.plot(range(1, 12), acc_values, marker='o')
    plt.xlabel("k value")
    plt.ylabel("Accuracy")
    plt.show()

    cm, p, r, f = a12_confusion_and_metrics(y_test, y_test_pred)
    print("A12:", cm, p, r, f)

    print("A13:", a13_metrics_from_confusion(cm))

    matrix_pred = a14_matrix_inversion_classifier(X_train, y_train, X_test)
    print("A14 Accuracy:", accuracy_score(y_test, matrix_pred))


main()
