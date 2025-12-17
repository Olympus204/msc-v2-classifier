import csv
import random
from data_pipeline.clean_raw import clean_row

def load_raw_data(path, max_rows=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append(row)
            if max_rows and i >= max_rows:
                break
    return rows

def pick_random_row(rows):
    return random.choice(rows)

def test_random_row(rows):
    raw = pick_random_row(rows)
    print("\n=== RAW ROW ===")
    for k, v in raw.items():
        print(f"{k}: {repr(v)}")

    print("\n=== CLEANED ===")
    try:
        cleaned = clean_row(raw)
        print(cleaned)
    except Exception as e:
        print("ERROR during cleaning:", e)

if __name__ == "__main__":
    path = r"C:\Projects\MSCV2\data\master_dataset.csv"   # change this
    rows = load_raw_data(path, max_rows=5000)       # load first 5k rows
    test_random_row(rows)
