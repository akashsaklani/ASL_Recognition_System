import pandas as pd
import csv

# INPUT FILE
input_file = "dataset.csv"

# OUTPUT FILE
output_file = "clean_dataset.csv"

# Required number of columns
EXPECTED_COLUMNS = 43

cleaned_rows = []

with open(input_file, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)

    for row in reader:

        # Agar columns zyada hain
        if len(row) > EXPECTED_COLUMNS:
            row = row[:EXPECTED_COLUMNS]

        # Agar columns kam hain
        elif len(row) < EXPECTED_COLUMNS:
            row += [''] * (EXPECTED_COLUMNS - len(row))

        cleaned_rows.append(row)

# Save cleaned dataset
df = pd.DataFrame(cleaned_rows)
df.to_csv(output_file, index=False, header=False)

print("Dataset successfully cleaned!")
print(f"Saved as: {output_file}")