import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
#i did not implement seaborn as its very buggy with python 3.13

def A1_purchase_analysis(file): #I formed matrix X from purchase quantities, computed its rank and dimensionality, and calculated the cost vector using pseudo-inverse.
    df = pd.read_excel(file, sheet_name="Purchase data")
    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].to_numpy()
    Y = df["Payment (Rs)"].to_numpy()
    dimension = X.shape[1]
    vectors = X.shape[0]
    rank = np.linalg.matrix_rank(X)
    cost = np.linalg.pinv(X) @ Y
    return dimension, vectors, rank, cost

def A2_rich_poor(file): #This is rule-based classification using a threshold.
    df = pd.read_excel(file, sheet_name="Purchase data")
    df["Class"] = np.where(df["Payment (Rs)"] > 200, "RICH", "POOR")
    return df

def A3_stock_stats(file): #I computed descriptive statistics and probabilities using filtering
    df = pd.read_excel(file, sheet_name="IRCTC Stock Price")
    price = df["Price"].to_numpy()
    mean_pkg = df["Price"].mean()
    var_pkg = df["Price"].var()
    mean_manual = price.sum() / len(price)
    var_manual = ((price - mean_manual) ** 2).sum() / len(price)
    wed_mean = df[df["Day"] == "Wednesday"]["Price"].mean()
    april_mean = df[df["Month"] == "April"]["Price"].mean()
    loss_prob = (df["Chg%"] < 0).sum() / len(df)

    wed = df[df["Day"] == "Wednesday"]
    profit_wed = (wed["Chg%"] > 0).sum() / len(wed) if len(wed) > 0 else 0

    return mean_pkg, var_pkg, mean_manual, var_manual, wed_mean, april_mean, loss_prob, profit_wed

def A4_data_exploration(file): #This helps understand attribute types and missing data
    df = pd.read_excel(file, sheet_name="thyroid0387_UCI")
    types = df.dtypes
    missing = df.isnull().sum()
    return types, missing

def A5_jaccard_smc(v1, v2): #These are binary similarity measures
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))
    jc = f11 / (f11 + f10 + f01)
    smc = (f11 + f00) / (f11 + f10 + f01 + f00)
    return jc, smc

def A6_cosine_similarity(v1, v2): #Cosine similarity measures angular similarity
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def A7_similarity_heatmaps(data): #I visualized similarity matrices using heatmaps directly with matplotlib with the help of AI since seaborn was returning constant errors in mysystem due to version incompatibility
    import os # i had to separately import os to import the output graphs as images to a selected directory, without which i couldnt locate the output file
    

    print("A7_similarity_heatmaps called, data shape:", data.shape)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    n = data.shape[0]
    jc = np.zeros((n, n))
    smc = np.zeros((n, n))
    cos = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            jc[i, j], smc[i, j] = A5_jaccard_smc(data[i], data[j])
            cos[i, j] = A6_cosine_similarity(data[i], data[j])

    plt.figure()
    plt.imshow(jc)
    plt.colorbar()
    plt.title("Jaccard Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "A7_Jaccard_Heatmap.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.imshow(smc)
    plt.colorbar()
    plt.title("SMC Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "A7_SMC_Heatmap.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.imshow(cos)
    plt.colorbar()
    plt.title("Cosine Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "A7_Cosine_Heatmap.png"), dpi=300)
    plt.close()

    print("Heatmaps saved in:", output_dir)



def A8_imputation(file): #Different imputation strategies for numeric and categorical data.
    df = pd.read_excel(file, sheet_name="thyroid0387_UCI")
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

def A9_normalization(df): #Normalization brings all features to the same scale.
    num = df.select_dtypes(include=["int64", "float64"]).columns
    df[num] = (df[num] - df[num].min()) / (df[num].max() - df[num].min())
    return df

def main():
    print("this code was developed by Rohan U (bl.sc.u4aie24043) for 22aie213")
    file = r"C:\Users\rohan\Downloads\ml assignments\Lab Session Data.xlsx"

    dim, vecs, rank, cost = A1_purchase_analysis(file)
    print("A1 Purchase Analysis")
    print("Dimension:", dim)
    print("Number of vectors:", vecs)
    print("Rank of matrix:", rank)
    print("Cost vector:", cost)
    print("-" * 50)

    df_class = A2_rich_poor(file)
    print("A2 RICH / POOR Classification (first 5 rows)")
    print(df_class[["Payment (Rs)", "Class"]].head())
    print("-" * 50)

    mean_pkg, var_pkg, mean_man, var_man, wed_mean, april_mean, loss_prob, profit_wed = A3_stock_stats(file)
    print("A3 Stock Statistics")
    print("Mean (package):", mean_pkg)
    print("Variance (package):", var_pkg)
    print("Mean (manual):", mean_man)
    print("Variance (manual):", var_man)
    print("Wednesday mean price:", wed_mean)
    print("April mean price:", april_mean)
    print("Probability of loss:", loss_prob)
    print("Probability of profit on Wednesday:", profit_wed)
    print("-" * 50)

    types, missing = A4_data_exploration(file)
    print("A4 Thyroid Data Exploration")
    print("Data types:")
    print(types)
    print("Missing values:")
    print(missing)
    print("-" * 50)

    df = A8_imputation(file)
    df = A9_normalization(df)
    print("A8 + A9 After Imputation and Normalization (first 5 rows)")
    print(df.head())
    print("-" * 50)

    num_df = df.select_dtypes(include=["int64", "float64"]).iloc[:20]
    binary = (num_df > 0).astype(int).to_numpy()

    A7_similarity_heatmaps(binary)
    print("A7 Heatmaps saved as image files:")
    print(" - A7_Jaccard_Heatmap.png")
    print(" - A7_SMC_Heatmap.png")
    print(" - A7_Cosine_Heatmap.png")

main()
