"""
zero_shot_bart.py

This script performs zero-shot sentiment classification on a Hebrew dataset
using the English 'facebook/bart-large-mnli' model. It uses tqdm for progress tracking
and saves the final results to a new CSV file.

Author: Tsofiya Shalev
Date: July 2025
"""

from transformers import pipeline
import pandas as pd
from tqdm import tqdm

# Load zero-shot classification model
print("✅ Loading facebook/bart-large-mnli zero-shot classification model...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print("✅ Model loaded successfully.")

# Load dataset
df = pd.read_csv("dataset.csv")
print(f"✅ Dataset loaded with {len(df)} rows.")

# Define candidate labels (English)
candidate_labels = ["positive", "negative"]

# Define prediction function with tqdm for progress tracking
def predict_sentiment(text):
    result = classifier(text, candidate_labels)
    return result['labels'][0]

# Apply with progress bar
print("🚀 Running zero-shot classification with progress bar...")
tqdm.pandas()
df["zero_shot_prediction_bart"] = df["text"].astype(str).progress_apply(predict_sentiment)

# Save results to new CSV
output_file = "dataset_zero_shot_bart.csv"
df.to_csv(output_file, index=False)
print(f"✅ Zero-shot classification completed and saved to {output_file}.")
