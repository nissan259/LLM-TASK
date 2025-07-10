import pandas as pd

# Load the dataset
df = pd.read_csv("token_train.tsv", sep="\t")

# Display initial info
print("✅ Original dataset loaded.")
print(df.head())

# Rename columns to English for clarity
df = df.rename(columns={"טקסט": "text", "0": "label"})

# Map labels to sentiment names
# ⚠️ בדקי במסמך המקור אילו מספרים מייצגים חיובי / שלילי / נייטרלי
# כאן אני מניח לדוגמא: 0=negative, 1=positive
label_map = {
    0: "negative",
    1: "positive"
}

df["sentiment"] = df["label"].map(label_map)

# Save as dataset.csv
df.to_csv("dataset.csv", index=False, encoding="utf-8-sig")

print("✅ dataset.csv saved successfully with columns: text, label, sentiment")
print(df.head())
