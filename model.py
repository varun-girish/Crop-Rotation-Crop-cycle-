# 🔍 model.py — Only model training, evaluation, and saving (no Flask)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# 1. Load and clean dataset
DATA_PATH = "/Users/varungirish/docs/college/AI_ML/project/karnataka.csv"
df = pd.read_csv(DATA_PATH)


#display head and tail of the dataset
print("Dataset Head:\n", df.head())
print("\nDataset Tail:\n", df.tail())

#display the description of the dataset
print("\nDataset Description:\n", df.describe())

# display the columns of the dataset
print("\nDataset Columns:\n", df.columns)

#display the null values in the dataset
print("\nNull Values in Dataset:\n", df.isnull().sum())

# display the shape of the dataset
print("\nDataset Shape:\n", df.shape)

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df.rename(columns={"yeilds": "yield"}, inplace=True)
df.drop(columns=["area"], inplace=True, errors="ignore")
df.dropna(inplace=True)
#dataset shape after cleaning
print("\nDataset Shape after Cleaning:\n", df.shape)
# 2. Encode categorical columns
categorical_cols = ['location', 'soil_type', 'irrigation', 'season', 'crops']
encoders = {}
for col in categorical_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col])
    encoders[col] = enc

# 3. Split features and label
X = df.drop(columns=['crops'])
y = df['crops']
feature_order = X.columns.tolist()

# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate
print("\n Model Evaluation Results:")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
#r2 score of the model
r2_score = model.score(X_test, y_test)
print(f"R2 Score: {r2_score:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoders['crops'].classes_))

#mse of the model
mse = np.mean((y_test - y_pred) ** 2)
print(f"\nMean Squared Error: {mse:.2f}")

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=encoders['crops'].classes_,
            yticklabels=encoders['crops'].classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 9. Feature Importances
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_order)
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()


#  Save components for Flask app
joblib.dump(model, "model1.pkl")
joblib.dump(scaler, "scaler1.pkl")
joblib.dump(encoders, "encoders1.pkl")
joblib.dump(feature_order, "feature_names1.pkl")

print("\n Model and preprocessing tools saved successfully!")