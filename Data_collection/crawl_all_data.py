import requests
import pandas as pd
import csv
import time
import os
import re

# ======================
# CONFIG
# ======================

URL = "https://eduvale.in/eapcet-college-wise-allotment/fetch_results.php"
HEADERS = {"User-Agent": "Mozilla/5.0"}

YEARS = list(range(2015, 2025))

BRANCH_CODES = [
    "CIV", "CSE", "ECE", "EEE", "MECH",
    "IT", "CSM", "CSO", "AIM", "AIDS"
]

COLLEGE_FILE = "college_codes_clean.csv"
OUTPUT_FILE = "ts_eapcet_all_data.csv"
PROGRESS_FILE = "progress.txt"

DELAY = 0.4

# ======================
# LOAD COLLEGE CODES
# ======================

college_df = pd.read_csv(COLLEGE_FILE)
college_list = college_df.to_dict("records")

print(f"Loaded {len(college_list)} colleges")

# ======================
# LOAD PROGRESS
# ======================

done = set()
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        done = set(line.strip() for line in f)

# ======================
# CSV SETUP
# ======================

file_exists = os.path.exists(OUTPUT_FILE)

csv_file = open(OUTPUT_FILE, "a", newline="", encoding="utf-8")
writer = csv.writer(csv_file)

if not file_exists:
    writer.writerow([
        "Year",
        "College Code",
        "College Name",
        "Branch",
        "Roll No",
        "Rank",
        "Candidate Name",
        "Gender",
        "Region",
        "Category",
        "Seat Category"
    ])

# ======================
# HELPER: CLEAN COLLEGE NAME
# ======================

def clean_college_name(name):
    match = re.search(
        r"(.*?(COLLEGE|INSTITUTE|UNIVERSITY))",
        str(name),
        re.IGNORECASE
    )
    return match.group(1).strip() if match else str(name).strip()

# ======================
# MAIN LOOP
# ======================

try:
    for year in YEARS:
        for college in college_list:
            code = college["college_code"]

            # <<< FIX: sanitize college name at runtime
            name = clean_college_name(college["college_name"])

            for branch in BRANCH_CODES:
                key = f"{year}-{code}-{branch}"
                if key in done:
                    continue

                params = {
                    "year": year,
                    "college": code,
                    "branch": branch,
                    "draw": 1,
                    "start": 0,
                    "length": 1000
                }

                try:
                    r = requests.get(
                        URL,
                        params=params,
                        headers=HEADERS,
                        timeout=20
                    )
                    rows = r.json().get("data", [])
                except Exception:
                    rows = []

                if rows:
                    for row in rows:
                        writer.writerow([
                            year,
                            code,
                            name,
                            branch,
                            row.get("rollno"),
                            row.get("rank"),
                            row.get("cand_name"),
                            row.get("gender"),
                            row.get("region"),
                            row.get("category"),
                            row.get("seat_category")
                        ])

                    try:
                        csv_file.flush()
                    except PermissionError:
                        pass

                    print(f"✓ {year} | {code} | {branch} | {len(rows)} rows")

                with open(PROGRESS_FILE, "a") as f:
                    f.write(key + "\n")
                done.add(key)

                time.sleep(DELAY)

finally:
    csv_file.close()

print("🎉 ALL DATA DOWNLOADED SUCCESSFULLY")
