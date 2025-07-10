import pandas as pd
from transformers import pipeline

# ✅ טען את המודל (heBERT) והגדר שהריצה תתבצע על CPU
print("✅ Loading heBERT model...")
classifier = pipeline(
    "sentiment-analysis",
    model="avichr/heBERT",
    tokenizer="avichr/heBERT",
    device=-1  # CPU
)
print("✅ Model loaded successfully.")

# ✅ טען את הדאטסט עם header = 0 כי עכשיו הוספת כותרות
df = pd.read_csv("dataset.csv", header=0)

# ✅ הדפס שם עמודות לבדיקה
print("🔎 Columns:", df.columns)

# ✅ פונקציה לחיזוי סנטימנט עבור טקסט בודד
def predict_sentiment(text):
    try:
        result = classifier(text)
        return result[0]["label"]
    except Exception as e:
        print(f"Error predicting sentiment for text: {text}\n{e}")
        return "error"

# ✅ צור עמודת תחזית חדשה עם zero-shot
print("✅ Running zero-shot sentiment evaluation...")
df["zero_shot_prediction"] = df["text"].astype(str).apply(predict_sentiment)

# ✅ שמור את הקובץ עם התוצאות
output_path = "dataset_zero_shot.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"✅ Zero-shot predictions saved to {output_path}.")
