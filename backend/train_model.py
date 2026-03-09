import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Loading dataset...")

df = pd.read_csv("dataset/ts_eapcet_all_data.csv", low_memory=False)

print("Cleaning Rank column...")
df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
df = df.dropna(subset=["Rank"])

print("Dropping unnecessary columns...")
df = df.drop(columns=["Candidate Name", "Roll No"], errors="ignore")

print("Creating target column...")
df["college_branch"] = df["College Name"] + " - " + df["Branch"]

# -----------------------------

# Reduce dataset size (important)

# -----------------------------

print("Sampling dataset to reduce memory usage...")
df = df.sample(120000, random_state=42)

print("Selecting features...")
features = ["Rank", "Gender", "Category", "Region"]
X = df[features].copy()

y = df["college_branch"]

print("Encoding categorical features...")

encoders = {}

for col in ["Gender", "Category", "Region"]:
    le = LabelEncoder()
    X.loc[:, col] = le.fit_transform(X[col])
    encoders[col] = le

print("Encoding target...")
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
X,
y_encoded,
test_size=0.2,
random_state=42
)

print("Training RandomForest model...")

model = RandomForestClassifier(
n_estimators=50,     # reduced from 150
max_depth=15,        # reduced depth
n_jobs=-1,
random_state=42
)

model.fit(X_train, y_train)

print("Evaluating model...")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

print("Saving model...")

joblib.dump(model, "model/eamcet_model.pkl")
joblib.dump(encoders, "model/encoder.pkl")
joblib.dump(target_encoder, "model/target_encoder.pkl")

print("Model training completed and saved!")
