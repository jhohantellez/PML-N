import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, auc, roc_curve
from sklearn.preprocessing import label_binarize

data = pd.read_excel("UNGRD_Scaled.xlsx")

data["VULNERABILITY_LEVEL"] = data["VULNERABILITY_LEVEL"].replace("Critical", "High")

print(data.head())
print("\nClass distribution (raw records):")
print(data["VULNERABILITY_LEVEL"].value_counts())

FEATURE_COLS = [
    "DECEASED", "INJURED", "PEOPLE", "FAMILIES",
    "DESTROYED HOUSES", "DAMAGED HOUSES", "DAMAGED ROADS",
    "EDUCATIONAL CENTERS", "HECTARES", "SEVERITY_INDEX",
    "MUNICIPALITY_EVENT_COUNT", "EVENT_DIVERSITY",
    "AVG_SEVERITY_MUNICIPALITY", "RAINY_SEASON", "VULNERABILITY_INDEX",
]
SUM_COLS = ["DECEASED", "INJURED", "PEOPLE", "FAMILIES",
            "DESTROYED HOUSES", "DAMAGED HOUSES", "DAMAGED ROADS",
            "EDUCATIONAL CENTERS", "HECTARES"]
MAX_COLS = ["MUNICIPALITY_EVENT_COUNT", "EVENT_DIVERSITY"]
MEAN_COLS = ["SEVERITY_INDEX", "AVG_SEVERITY_MUNICIPALITY",
             "RAINY_SEASON", "VULNERABILITY_INDEX"]

agg_dict = (
    {c: "sum"  for c in SUM_COLS}  |
    {c: "max"  for c in MAX_COLS}  |
    {c: "mean" for c in MEAN_COLS} |
    {"VULNERABILITY_LEVEL": lambda x: x.mode()[0]}
)

muni_df = data.groupby("MUNICIPALITY").agg(agg_dict).reset_index()
print(f"\nMunicipalities: {len(muni_df)}")
print(muni_df["VULNERABILITY_LEVEL"].value_counts())

CLASS_ORDER = ["Low", "Medium", "High"]

x = muni_df[FEATURE_COLS]
y = muni_df["VULNERABILITY_LEVEL"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled  = scaler.transform(x_test)

logistic_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
logistic_model.fit(x_train_scaled, y_train)

y_pred = logistic_model.predict(x_test_scaled)

conf_matrix = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Climate Vulnerability Classification")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

print(classification_report(y_test, y_pred, target_names=CLASS_ORDER))

accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")


y_test_bin = label_binarize(y_test, classes=CLASS_ORDER)
y_prob_all  = logistic_model.predict_proba(x_test_scaled)

COLORS = {"Low": "#22d3a0", "Medium": "#f5a623", "High": "#e84545"}

plt.figure(figsize=(8, 6))
for i, cls in enumerate(CLASS_ORDER):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob_all[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=COLORS[cls], lw=2,
             label=f"{cls} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (One-vs-Rest) – Climate Vulnerability")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.show()

coef_abs = np.abs(logistic_model.coef_).mean(axis=0)
feat_series = pd.Series(coef_abs, index=FEATURE_COLS).sort_values(ascending=True)

colors = [COLORS["High"] if v >= feat_series.median() else COLORS["Low"]
          for v in feat_series]

plt.figure(figsize=(9, 6))
plt.barh(feat_series.index, feat_series.values, color=colors)
plt.xlabel("Mean |Coefficient|")
plt.title("Feature Importance – Logistic Regression")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

def pre_resultado(datos_entrada: dict):
    df_input = pd.DataFrame([datos_entrada])
    df_input = df_input.reindex(columns=FEATURE_COLS, fill_value=0)
    datos_escalados = scaler.transform(df_input)
    prediccion   = logistic_model.predict(datos_escalados)[0]
    probabilidad = logistic_model.predict_proba(datos_escalados)[0]
    return prediccion, dict(zip(CLASS_ORDER, probabilidad.tolist()))

results = muni_df[["MUNICIPALITY"]].copy()
results["TRUE_LABEL"]      = y.values
results["PREDICTED_LABEL"] = logistic_model.predict(scaler.transform(x))
prob_df = pd.DataFrame(
    logistic_model.predict_proba(scaler.transform(x)),
    columns=[f"PROB_{c.upper()}" for c in CLASS_ORDER],
)
results = pd.concat([results.reset_index(drop=True), prob_df], axis=1)
results.to_csv("municipality_vulnerability_predictions.csv", index=False)
print("Predictions saved → municipality_vulnerability_predictions.csv")

if __name__ == "__main__":
    print("ready model")