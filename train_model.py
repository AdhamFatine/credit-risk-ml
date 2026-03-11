# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1️⃣ Load dataset
data = pd.read_csv("data/credit_dataset.csv")

# 2️⃣ Select features li bghiti + target
features = [
    "status_account", "month_duration", "credit_history", "purpose",
    "credit_amount", "status_savings", "years_employment",
    "payment_to_income_ratio", "status_and_sex", "secondary_obligor",
    "residence_since", "collateral", "age", "other_installment_plans",
    "housing", "n_credits", "job", "n_guarantors", "telephone",
    "is_foreign_worker"
]

target = "target"

data_subset = data[features + [target]]

# 3️⃣ Preprocessing
# Convert target good/bad → 1/0
data_subset['target'] = data_subset['target'].map({'good':1, 'bad':0})

# Encode text columns
le = LabelEncoder()
for col in data_subset.select_dtypes(include='object').columns:
    if col != 'target':
        data_subset[col] = le.fit_transform(data_subset[col])

# 4️⃣ Split dataset
X = data_subset.drop("target", axis=1)
y = data_subset["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️⃣ Train modèle
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 6️⃣ Predict w Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 7️⃣ Save model
joblib.dump(model, "model/credit_model.pkl")
print("Model saved in model/credit_model.pkl")