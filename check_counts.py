import pandas as pd

df = pd.read_csv("test_queries.csv")

print("Total rows in CSV:", len(df))
print("Missing query_id:", df["query_id"].isna().sum())
print("Missing query_text:", df["query_text"].isna().sum())
print("Missing language:", df["language"].isna().sum())
print("Missing correct_faq_id:", df["correct_faq_id"].isna().sum())

# rows that are fully empty
print("Fully empty rows:", df.isna().all(axis=1).sum())

# rows that are missing any required field
required = ["query_id", "query_text", "language", "correct_faq_id"]
bad_rows = df[df[required].isna().any(axis=1)]
print("Rows missing required fields:", len(bad_rows))

# show bad rows (so you can see what's wrong)
if len(bad_rows) > 0:
    print("\nBad rows preview:")
    print(bad_rows.head(20))
