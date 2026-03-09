import joblib
import numpy as np
import pandas as pd
import re

# -----------------------------

# Load Model and Encoders

# -----------------------------
def clean_college_branch(text):

    import re

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # remove extra spaces
    text = " ".join(text.split())

    # remove duplicate words
    words = text.split()
    seen = set()
    clean_words = []

    for w in words:
        if w not in seen:
            clean_words.append(w)
            seen.add(w)

    text = " ".join(clean_words)

    return text.strip()
model = joblib.load("model/eamcet_model.pkl")
encoders = joblib.load("model/encoder.pkl")
target_encoder = joblib.load("model/target_encoder.pkl")

# -----------------------------

# Load Dataset for Cutoff Logic

# -----------------------------

df = pd.read_csv("dataset/ts_eapcet_all_data.csv", low_memory=False)
df["College Name"] = df["College Name"].apply(clean_college_branch)
# Calculate historical closing ranks

closing_ranks = (
df.groupby(["College Name", "Branch"])["Rank"]
.max()
.reset_index()
)

# -----------------------------

# Hybrid Classification

# -----------------------------

def classify_cutoff(student_rank, closing_rank):

    if student_rank <= closing_rank:
        return "Safe"

    elif student_rank <= closing_rank + 3000:
        return "Target"

    else:
        return "Dream"

# -----------------------------

# Prediction Function

# -----------------------------

def predict_colleges(rank, gender, category, region, top_k=5):

    gender = gender.lower()

    if gender in ["male", "m"]:
        gender = "M"
    elif gender in ["female", "f"]:
        gender = "F"

    gender_encoded = encoders["Gender"].transform([gender])[0]
    category_encoded = encoders["Category"].transform([category])[0]
    region_encoded = encoders["Region"].transform([region])[0]

    X_input = np.array([[rank, gender_encoded, category_encoded, region_encoded]])

    probabilities = model.predict_proba(X_input)[0]

    top_indices = np.argsort(probabilities)[-top_k:][::-1]

    predicted_results = []

    for idx in top_indices:

        college_branch = target_encoder.inverse_transform([idx])[0]

        college_branch = clean_college_branch(college_branch)

        probability = probabilities[idx]

        parts = college_branch.split("-")

        if len(parts) >= 2:
            college = parts[0].strip()
            branch = parts[1].strip()
        else:
            college = college_branch
            branch = ""

        row = closing_ranks[
            (closing_ranks["College Name"] == college) &
            (closing_ranks["Branch"] == branch)
        ]

        if len(row) > 0:
            closing_rank = row.iloc[0]["Rank"]
            category_label = classify_cutoff(rank, closing_rank)
        else:
            category_label = "Target"

        predicted_results.append({
            "college_branch": college_branch,
            "probability": round(float(probability),3),
            "category": category_label
        })

    # ------------------------------
    # Find ALL Possible Colleges
    # ------------------------------

    possible = closing_ranks[
        closing_ranks["Rank"] >= rank
    ]

    possible_colleges = []

    for _, row in possible.iterrows():

        college = row["College Name"]
        branch = row["Branch"]

        college_branch = clean_college_branch(f"{college} - {branch}")

        # remove numbers and extra cutoff values
        college_branch = re.sub(r'\d+', '', college_branch)

        # remove extra spaces
        college_branch = " ".join(college_branch.split())

        possible_colleges.append(college_branch)

    possible_colleges = list(set(possible_colleges))[:30]

    return {
        "predicted": predicted_results,
        "possible": possible_colleges
    }
# -----------------------------

# Test Prediction (CLI)

# -----------------------------

if __name__ == "__main__":

    results = predict_colleges(
        rank=5000,
        gender="Male",
        category="BC_B",
        region="OU"
    )

    print("\nTop Predicted Colleges:\n")

    for r in results:
        print(r)
