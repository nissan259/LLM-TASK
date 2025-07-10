# =====================================
# ✅ Zero-Shot BART Evaluation Analysis Script (Final Version)
# =====================================

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 📥 1. Load the dataset
# ===============================
df = pd.read_csv("dataset_zero_shot_bart.csv")
print("✅ Dataset loaded successfully.")

# Display column names for verification
print("📄 Columns:", df.columns)

# ===============================
# 🧹 2. Data Cleaning: Filter valid sentiments only
# ===============================
# Keep only rows where label_sentiment is 'positive' or 'negative'
df = df[df["label_sentiment"].isin(["positive", "negative"])]

# ===============================
# 📊 3. Basic summary statistics
# ===============================
total = len(df)
correct = (df["label_sentiment"] == df["zero_shot_prediction_bart"]).sum()
accuracy = correct / total * 100

print(f"✅ Total samples: {total}")
print(f"✅ Correct predictions: {correct}")
print(f"✅ Accuracy: {accuracy:.2f}%")

# ===============================
# 🔢 4. Count of true labels and predictions
# ===============================
print("\n🔹 True Sentiment Counts:")
print(df["label_sentiment"].value_counts())

print("\n🔹 Model Prediction Counts:")
print(df["zero_shot_prediction_bart"].value_counts())

# ===============================
# ❌ 5. Analyze incorrect predictions (errors)
# ===============================
errors = df[df["label_sentiment"] != df["zero_shot_prediction_bart"]]
errors.to_csv("zero_shot_bart_errors.csv", index=False)
print(f"❌ Total Errors: {len(errors)} (saved to zero_shot_bart_errors.csv)")

# Display first few error examples
print("\n🔍 Example Errors:")
print(errors.head(10))

# ===============================
# 📊 6. Classification Report
# ===============================
print("\n🔹 Classification Report:")
print(classification_report(df["label_sentiment"], df["zero_shot_prediction_bart"]))

# ===============================
# 🟦 7. Confusion Matrix
# ===============================
cm = confusion_matrix(df["label_sentiment"], df["zero_shot_prediction_bart"], labels=["positive", "negative"])
cm_df = pd.DataFrame(cm, index=["True Positive", "True Negative"], columns=["Pred Positive", "Pred Negative"])

plt.figure(figsize=(6,4))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Zero Shot BART")
plt.savefig("confusion_matrix_zero_shot_bart.png")
plt.show()

print("✅ Confusion Matrix saved as confusion_matrix_zero_shot_bart.png")

# ===============================
# 💾 8. Save overall summary
# ===============================
summary = {
    "Total Samples": total,
    "Correct Predictions": correct,
    "Accuracy (%)": accuracy,
    "Errors": len(errors)
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv("zero_shot_bart_summary.csv", index=False)

print("✅ Summary saved as zero_shot_bart_summary.csv")
