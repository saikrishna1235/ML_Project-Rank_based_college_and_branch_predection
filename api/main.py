from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from backend.predict import predict_colleges
import pandas as pd
import sys
import os
import re
# --------- FastAPI APP ---------

app = FastAPI()

# --------- Enable CORS ---------

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

# --------- Load Dataset for Insights ---------

df = pd.read_csv("dataset/ts_eapcet_all_data.csv", low_memory=False)

# --------- Import Prediction Function ---------

sys.path.append(os.path.abspath("backend"))




# --------- Request Model ---------

class StudentInput(BaseModel):
    rank: int
    gender: str
    category: str
    region: str

# --------- Home Route ---------

@app.get("/")
def home():
    return {"message": "TS EAMCET College Predictor API Running"}

# --------- Prediction API ---------

@app.post("/predict")
def predict(data: StudentInput):

    print("Received data:", data.dict())

    results = predict_colleges(
        rank=data.rank,
        gender=data.gender,
        category=data.category,
        region=data.region
    )

    return results

# --------- Insights API ---------

@app.get("/insights")
def insights():

    total_colleges = df["College Name"].nunique()
    total_branches = df["Branch"].nunique()
    total_records = len(df)

    top_branches = (
        df["Branch"]
        .value_counts()
        .head(5)
        .to_dict()
    )

    return {
        "total_colleges": total_colleges,
        "total_branches": total_branches,
        "records": total_records,
        "top_branches": top_branches
    }

@app.get("/trend")
def cutoff_trend(college: str, branch: str):

    data = df[
        (df["College Name"].str.contains(college, case=False)) &
        (df["Branch"].str.contains(branch, case=False))
    ]

    # get closing rank per year
    trend = (
        data.groupby("Year")["Rank"]
        .max()
        .reset_index()
        .sort_values("Year")
    )

    return trend.to_dict(orient="records")
@app.get("/college_search")
def college_search(college:str):

    data = df[
        df["College Name"].str.contains(college, case=False)
    ]

    results = (
        data[["College Name","Branch","Rank","Year"]]
        .drop_duplicates()
        .sort_values(["Branch","Year"])
    )

    return results.to_dict(orient="records")
@app.get("/college_suggestions")
def college_suggestions(query: str):

    data = df[
        df["College Name"].str.contains(query, case=False, na=False)
    ]["College Name"].drop_duplicates()

    cleaned = []

    for name in data:

        # remove numbers
        name = re.sub(r'\d+', '', name)

        # remove NA
        name = re.sub(r'\bNA\b', '', name)

        # remove repeated spaces
        name = re.sub(r'\s+', ' ', name).strip()

        # remove branch words
        name = re.sub(r'\b(CSE|ECE|EEE|CIV|MECH|IT|CIVIL ENGINEERING)\b', '', name)

        cleaned.append(name)

    # remove duplicates again
    cleaned = list(set(cleaned))

    return sorted(cleaned)[:10]
@app.get("/compare")
def compare(college1: str, college2: str):

    def get_college_data(college):

        data = df[df["College Name"].str.contains(college, case=False, na=False)]

        if data.empty:
            return {
                "college": college,
                "branches": [],
                "trend": []
            }

        branches = data["Branch"].dropna().unique().tolist()

        # Use Rank column
        trend = (
            data.groupby("Year")["Rank"]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        )

        return {
            "college": college,
            "branches": branches,
            "trend": trend
        }

    return {
        "college1": get_college_data(college1),
        "college2": get_college_data(college2)
    }
@app.get("/college_list")
def college_list():

    colleges = df["College Name"].dropna().unique()

    clean = []

    for c in colleges:

        c = re.sub(r'\d+', '', c)
        c = re.sub(r'\bNA\b', '', c)
        c = re.sub(r'\s+', ' ', c).strip()

        clean.append(c)

    clean = list(set(clean))

    return sorted(clean)
@app.post("/counseling_list")
def counseling_list(data: InputData):

    results = predict_colleges(
        data.rank,
        data.gender,
        data.category,
        data.region
    )

    counseling = []

    for r in results:
        counseling.append({
            "college": r
        })

    return {"list": counseling}