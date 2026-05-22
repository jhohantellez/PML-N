import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import base64
from io import BytesIO

# -------------------- DATASETS --------------------
def get_dataset():
    df_scaled = pd.read_excel("UNGRD_Scaled.xlsx")
    df_original = pd.read_excel("Emergencias_UNGRD_Translated.xlsx")

    df_scaled.columns = df_scaled.columns.str.upper().str.strip()
    df_original.columns = df_original.columns.str.upper().str.strip()

    features = [
        "MUNICIPALITY_EVENT_COUNT",
        "EVENT_DIVERSITY",
        "RAINY_SEASON",
        "YEAR",
        "MONTH",
        "DEPARTMENT",
        "EVENT"
    ]

    X = df_scaled[features].dropna()

    # Sincronizar df_original con los índices válidos de X
    df_original = df_original.loc[X.index].reset_index(drop=True)
    X = X.reset_index(drop=True)

    return X, df_original


# -------------------- KMEANS --------------------
def ClimateKMeans(k=3):
    X, df_original = get_dataset()

    # -------------------- SPLIT --------------------
    X_train, X_test, idx_train, idx_test = train_test_split(
        X,
        X.index,
        test_size=0.2,
        random_state=42
    )

    # -------------------- SCALER --------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------- MODEL --------------------
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        max_iter=300,
        n_init=10,
        random_state=42
    )

    # -------------------- TRAINING --------------------
    labels = model.fit_predict(X_train_scaled)

    # -------------------- VALIDATION --------------------
    score = silhouette_score(X_train_scaled, labels)

    # -------------------- INERTIA --------------------
    inertia = model.inertia_

    # -------------------- PREDICTIONS --------------------
    predictions = model.predict(X_test_scaled)

    # -------------------- RESULTS TABLE --------------------
    results = X_test.copy().reset_index(drop=True)

    # Reemplazar DEPARTMENT, MUNICIPALITY, EVENT con nombres legibles
    for col in ["DEPARTMENT", "MUNICIPALITY", "EVENT"]:
        if col in df_original.columns:
            results[col] = df_original.iloc[idx_test.tolist()][col].values

    results["CLUSTER"] = predictions
    print("\nSample Predictions:")
    print(results.head(10))
    sampled = results.head(20).to_dict(orient="records")

    # -------------------- SUMMARY --------------------
    summary = {}
    for label in predictions:
        label = int(label)
        summary[label] = summary.get(label, 0) + 1

    # -------------------- CENTERS --------------------
    centers = model.cluster_centers_.tolist()


    # -------------------- ELBOW METHOD --------------------
    elbow = []

    for i in range(1, 11):

        km = KMeans(
            n_clusters=i,
            init="k-means++",
            max_iter=300,
            n_init=10,
            random_state=42
        )

        km.fit(X_train_scaled)

        elbow.append(km.inertia_)

    # -------------------- ELBOW GRAPH --------------------
    plt.figure(figsize=(7,5))

    plt.plot(range(1,11), elbow, marker='o')

    plt.title("Elbow Method")

    plt.xlabel("Number of Clusters (K)")

    plt.ylabel("Inertia")

    buffer2 = BytesIO()

    plt.savefig(buffer2, format="png")

    buffer2.seek(0)

    image_png2 = buffer2.getvalue()

    buffer2.close()

    elbow_graph = base64.b64encode(image_png2).decode("utf-8")

    plt.close()

    # -------------------- GRAPH --------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_train_scaled[:, 0],
        X_train_scaled[:, 1],
        c=labels,
        cmap="YlGn"
    )
    plt.title("Climate Vulnerability Clusters")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graph = base64.b64encode(image_png).decode("utf-8")
    plt.close()

    return {
    "summary": summary,
    "centers": centers,
    "results": sampled,
    "graph": graph,
    "score": round(score, 3),
    "inertia": round(inertia, 2),
    "elbow_graph": elbow_graph
}

# -------------------- TEST EXECUTION --------------------
if __name__ == "__main__":

    print("Running Climate K-Means Model...\n")

    info = ClimateKMeans(3)

    print("\nSilhouette Score:")
    print(info["score"])

    print("\nInertia:")
    print(info["inertia"])

    print("\nCluster Summary:")
    print(info["summary"])

    print("\nModel executed successfully.")
