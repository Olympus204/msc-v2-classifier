import csv
import sys
csv.field_size_limit(sys.maxsize)

INFILE  = "data/raw_harvest.csv"
OUTFILE = "data/raw_harvest_rewritten.csv"

with open(INFILE, newline="", encoding="utf-8", errors="ignore") as fin, \
     open(OUTFILE, "w", newline="", encoding="utf-8") as fout:

    reader = csv.reader(fin)
    writer = csv.writer(fout)

    # Write clean header
    writer.writerow([
        "id",
        "doi",
        "msc",
        "keywords",
        "title",
        "text",
        "refs"
    ])

    header = next(reader)  # skip original header

    for i, row in enumerate(reader, start=1):
        # Defensive: some rows may be shorter or longer
        if len(row) < 6:
            continue

        try:
            id_   = row[0]
            doi   = row[1]
            msc   = row[2]
            kw    = row[3]
            title = row[4]
            text  = row[5] if len(row) > 5 else ""
            refs  = row[6] if len(row) > 6 else ""

            writer.writerow([
                id_,
                doi,
                msc,
                kw,
                title,
                text,
                refs
            ])

        except Exception as e:
            print(f"[SKIP] row {i}: {e}")
