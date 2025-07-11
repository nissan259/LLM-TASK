"""
גרסה פשוטה של Fine-tuning
לבדיקה מהירה
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import time

class SimpleDataset:
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # טוקניזציה
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

def main():
    print("🚀 Simple Fine-tuning Test")
    
    try:
        # טען נתונים
        df = pd.read_csv("dataset_fixed.csv")
        df = df.dropna(subset=['text', 'label_sentiment'])
        df = df[df['label_sentiment'].isin(['negative', 'positive'])]
        
        # המר תוויות
        label_map = {'negative': 0, 'positive': 1}
        df['label_num'] = df['label_sentiment'].map(label_map)
        
        # חלק לטריין/טסט
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            df['text'].tolist(), df['label_num'].tolist(),
            test_size=0.2, random_state=42, stratify=df['label_num']
        )
        
        # קח דגימה קטנה לבדיקה
        train_texts = train_texts[:1000]
        train_labels = train_labels[:1000]
        test_texts = test_texts[:200]
        test_labels = test_labels[:200]
        
        print(f"✅ Using {len(train_texts)} train, {len(test_texts)} test samples")
        
        # טען מודל
        model_name = "avichr/heBERT"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        
        print("✅ Model loaded")
        
        # צור datasets
        train_dataset = SimpleDataset(train_texts, train_labels, tokenizer)
        test_dataset = SimpleDataset(test_texts, test_labels, tokenizer)
        
        # הגדרות אימון פשוטות
        training_args = TrainingArguments(
            output_dir="./simple_finetuned",
            num_train_epochs=1,  # רק epoch אחד לבדיקה
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs_simple",
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
        )
        
        # יצור trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
        
        print("🚀 Starting training...")
        start_time = time.time()
        
        # אמן
        trainer.train()
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds")
        
        # שמור
        trainer.save_model("./simple_finetuned")
        tokenizer.save_pretrained("./simple_finetuned")
        
        # העריך
        eval_result = trainer.evaluate()
        print(f"📊 Evaluation accuracy: {eval_result['eval_accuracy']:.4f}")
        
        # שמור תוצאות
        results_df = pd.DataFrame([{
            'method': 'Simple_Fine_tuning',
            'accuracy': eval_result['eval_accuracy'],
            'training_time': training_time,
            'train_samples': len(train_texts),
            'test_samples': len(test_texts)
        }])
        
        results_df.to_csv("simple_fine_tuning_results.csv", index=False)
        
        print(f"✅ Results saved to simple_fine_tuning_results.csv")
        print(f"🎯 Final accuracy: {eval_result['eval_accuracy']*100:.2f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
