import csv
import json
import sys
csv.field_size_limit(sys.maxsize)
from data_pipeline.clean_raw import clean_row

def process_csv(infile, outfile, encoding="utf-8"):
    
    total = 0
    cleaned = 0
    skipped = 0

    with open(infile, "r", encoding=encoding, errors="ignore", newline="") as f_in, \
        open(outfile, "w", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        print("COLUMNS:", reader.fieldnames)



        for row in reader:
            total += 1

            msc_val = row.get("msc", None)
            if total < 5:
                print("MSC RAW TYPE:", type(msc_val))
                print("MSC RAW VALUE:", str(msc_val)[:200])

            try:
                clean = clean_row(row)
            except Exception as e:
                skipped += 1
                print(f"[ERROR] Row {total} failed: {e}", file=sys.stderr)
                continue

            if clean is None:
                skipped += 1
                continue

            f_out.write(json.dumps(clean, ensure_ascii=False)+ "\n")
            cleaned += 1

            if total % 5000 == 0:
                print(f"[PROGRESS] {total} rows processed, {cleaned} cleaned, {skipped} skipped")
    print("\n=== DONE ===")
    print(f"Total rows: {total}")
    print(f"Cleaned:    {cleaned}")
    print(f"Skipped:    {skipped}")