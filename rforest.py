import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


df_original = pd.read_excel('Emergencias_UNGRD_Translated.xlsx')
df_original.columns = df_original.columns.str.upper().str.strip()


df_original['EVENT'] = (
    df_original['EVENT']    
    .astype(str)
    .str.upper()
    .str.strip()
)

event_translation = {
    'INUNDACIÓN': 'Flood',
    'INUNDACION': 'Flood',
    'INUNDACIÃ“N': 'Flood',
    'FLOOD': 'Flood',

    'DESLIZAMIENTO': 'Landslide',

    'VENDAVAL': 'Windstorm',
    'WINDSTORM': 'Windstorm',

    'INCENDIO': 'Fire',
    'VEGETATION COVER FIRE': 'Fire',

    'SEQUÍA': 'Drought',
    'DROUGHT': 'Drought',

    'TORMENTA': 'Storm',
    'TORMENTA ELECTRICA': 'Electrical Storm',

    'AVENIDA TORRENCIAL': 'Flash Flood',
    'CRECIENTE SUBITA': 'Flash Flood',
    'CRECIENTE SÚBITA': 'Flash Flood',

    'GRANIZADA': 'Hailstorm',

    'SISMO': 'Earthquake',

    'HELADA': 'Frost',

    'LLUVIAS': 'Heavy Rain',

    'QUEMA': 'Burning',

    'ACCIDENTE TRANSPORTE MARITIMO O FLUVIAL':
        'Maritime or River Transport Accident',

    'ACCIDENTE TRANSPORTE MARÃ\x8dTIMO O FLUVIAL':
        'Maritime or River Transport Accident',

    'CICLON TROPICAL: DEPRESION/TORMENTA/HURACAN':
        'Tropical Cyclone'
}
df_original['EVENT'] = df_original['EVENT'].replace(event_translation)

le_dept  = LabelEncoder()
le_event = LabelEncoder()

le_dept.fit(df_original['DEPARTMENT'])
le_event.fit(df_original['EVENT'])

print("Departments:", sorted(le_dept.classes_.tolist()))
print("Events:", sorted(le_event.classes_.tolist()))

with open('rf_le_dept.pkl', 'wb') as f:
    pickle.dump(le_dept, f)
with open('rf_le_event.pkl', 'wb') as f:
    pickle.dump(le_event, f)

df = pd.read_excel('UNGRD_Scaled.xlsx')

df_medium = df[df['VULNERABILITY_LEVEL'] == 'Medium']
df_medium_up = resample(df_medium, replace=True, n_samples=500, random_state=42)
df = pd.concat([df, df_medium_up])

features = [
    'MUNICIPALITY_EVENT_COUNT',
    'EVENT_DIVERSITY',
    'RAINY_SEASON',
    'YEAR',
    'MONTH',
    'DEPARTMENT',
    'EVENT'
]

X = df[features]
y = df['VULNERABILITY_LEVEL']

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]:,}")
print(f"Test samples:     {X_test.shape[0]:,}")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

y_score = rf.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


plt.figure(figsize=(8,6))

class_names = le.classes_

for i in range(3):
    plt.plot(
        fpr[i],
        tpr[i],
        label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})'
    )

plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')

plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.show()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

importances = pd.Series(rf.feature_importances_, index=features)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(8, 6))
importances.plot(kind='barh', color='steelblue')
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

print("\nTop Features:")
print(importances.sort_values(ascending=False))

print("\nDepartments:", sorted(le_dept.classes_.tolist()))
print("Events:",      sorted(le_event.classes_.tolist()))

with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

with open('rf_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print(sorted(df_original['EVENT'].unique()))

print("\nModel saved: rf_model.pkl")
print("Encoder saved: rf_encoder.pkl")