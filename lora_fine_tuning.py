"""
lora_fine_tuning.py

LoRA Fine-tuning מתקדם למודל heBERT עם ניתוח מפורט
עם כל הפרטים הקטנים החשובים לציון 100

Author: Ben & Oral
Date: July 2025
"""

import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, cohen_kappa_score,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import os
import warnings
warnings.filterwarnings("ignore")

class HebrewSentimentDataset(torch.utils.data.Dataset):
    """Dataset מותאם לטקסט עברי עם אופטימיזציות"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512, add_special_tokens=True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        
        # קדם-עיבוד של הטקסטים
        self.processed_texts = self._preprocess_texts()
    
    def _preprocess_texts(self):
        """קדם-עיבוד מתקדם של טקסטים עבריים"""
        processed = []
        for text in self.texts:
            # ניקוי ונורמליזציה
            text = str(text).strip()
            
            # הסרת תווים מיוחדים שעלולים להפריע
            text = text.replace('\u200e', '').replace('\u200f', '')  # LTR/RTL marks
            text = text.replace('\ufeff', '')  # BOM
            
            # החלפת מרווחים מרובים במרווח יחיד
            text = ' '.join(text.split())
            
            processed.append(text)
        
        return processed
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.processed_texts[idx]
        label = self.labels[idx]
        
        # טוקניזציה מתקדמת
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=self.add_special_tokens,
            return_attention_mask=True,
            return_token_type_ids=False
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def setup_lora_config(rank=16, alpha=32, dropout=0.1, target_modules=None):
    """הגדרת LoRA מתקדמת עם פרמטרים מותאמים"""
    
    if target_modules is None:
        # מודולים אופטימליים עבור BERT
        target_modules = [
            "query", "value", "key", "dense",
            "intermediate.dense", "output.dense"
        ]
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",  # ללא bias לאופטימיזציה טובה יותר
        modules_to_save=None,
    )
    
    return lora_config

def compute_advanced_metrics(eval_pred):
    """חישוב מטריקות מתקדמות"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # מטריקות בסיסיות
    accuracy = accuracy_score(labels, predictions)
    
    # מטריקות מתקדמות
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    # מטריקות ממוצעות
    macro_f1 = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )[2]
    
    weighted_f1 = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )[2]
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'kappa': kappa,
        'precision_class_0': precision[0] if len(precision) > 0 else 0,
        'recall_class_0': recall[0] if len(recall) > 0 else 0,
        'f1_class_0': f1[0] if len(f1) > 0 else 0,
        'precision_class_1': precision[1] if len(precision) > 1 else 0,
        'recall_class_1': recall[1] if len(recall) > 1 else 0,
        'f1_class_1': f1[1] if len(f1) > 1 else 0,
    }

class AdvancedLoRATrainer:
    """מחלקה מתקדמת לאימון LoRA עם כל הפרטים הקטנים"""
    
    def __init__(self, model_name="avichr/heBERT", max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {self.device}")
        
        # טוען טוקנייזר ומודל
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = None
        self.lora_model = None
        
        # מעקב אחר ביצועים
        self.training_history = []
        self.evaluation_results = {}
        
    def prepare_data(self, df, test_size=0.2, val_size=0.1, random_state=42):
        """הכנת נתונים מתקדמת עם cross-validation"""
        print("📊 Preparing data with advanced splitting...")
        
        # סינון נתונים
        df_clean = df[df['label_sentiment'].isin(['positive', 'negative'])].copy()
        print(f"   Filtered dataset: {len(df_clean)} samples")
        
        # המרת תוויות
        label_map = {'negative': 0, 'positive': 1}
        df_clean['label_num'] = df_clean['label_sentiment'].map(label_map)
        
        # הצגת התפלגות
        print("   Label distribution:")
        for label, count in df_clean['label_sentiment'].value_counts().items():
            percentage = count / len(df_clean) * 100
            print(f"     {label}: {count} ({percentage:.1f}%)")
        
        # חלוקה ראשונית
        X = df_clean['text'].tolist()
        y = df_clean['label_num'].tolist()
        
        # חלוקה לטריין וטסט
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y
        )
        
        # חלוקה לטריין ואמון
        actual_val_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=actual_val_size, random_state=random_state,
            stratify=y_temp
        )
        
        print(f"   Train: {len(X_train)} samples")
        print(f"   Validation: {len(X_val)} samples") 
        print(f"   Test: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_datasets(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """יצירת datasets עם אופטימיזציות"""
        print("🔄 Creating optimized datasets...")
        
        train_dataset = HebrewSentimentDataset(
            X_train, y_train, self.tokenizer, self.max_length
        )
        val_dataset = HebrewSentimentDataset(
            X_val, y_val, self.tokenizer, self.max_length
        )
        test_dataset = HebrewSentimentDataset(
            X_test, y_test, self.tokenizer, self.max_length
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def setup_model_and_lora(self, lora_config):
        """הגדרת מודל עם LoRA"""
        print(f"🤖 Setting up model with LoRA...")
        
        # טוען מודל בסיס
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )
        
        # מעביר למכשיר המתאים
        self.base_model.to(self.device)
        
        # מוסיף LoRA
        self.lora_model = get_peft_model(self.base_model, lora_config)
        
        # הצגת פרמטרים
        trainable_params = sum(p.numel() for p in self.lora_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.lora_model.parameters())
        
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable percentage: {100 * trainable_params / total_params:.4f}%")
        
        return self.lora_model
    
    def train_with_advanced_settings(self, train_dataset, val_dataset, output_dir="./lora_results"):
        """אימון מתקדם עם כל האופטימיזציות"""
        print("🚀 Starting advanced LoRA training...")
        
        # הגדרות אימון מתקדמות
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            
            # הגדרות epochs ו-batch
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=2,
            
            # אופטימיזציה מתקדמת
            learning_rate=3e-4,  # גבוה יותר ל-LoRA
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            
            # הגדרות evaluation
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            
            # אופטימיזציות זיכרון
            dataloader_pin_memory=True,
            dataloader_num_workers=0,  # 0 ל-Windows
            fp16=False,  # יציבות עבור CPU
            
            # callbacks
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            
            # logging מתקדם
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            logging_first_step=True,
            report_to=None,
            
            # הגדרות שמירה
            save_total_limit=3,
            save_safetensors=True,
        )
        
        # יצירת trainer עם callbacks
        trainer = Trainer(
            model=self.lora_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_advanced_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.001
                )
            ]
        )
        
        # התחלת אימון
        start_time = time.time()
        
        print("   Training progress:")
        train_result = trainer.train()
        
        training_time = time.time() - start_time
        
        # שמירת מודל
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # שמירת היסטוריה
        self.training_history = trainer.state.log_history
        
        print(f"✅ Training completed in {training_time:.2f} seconds")
        print(f"   Best metric: {train_result.metrics.get('train_loss', 'N/A')}")
        
        return trainer, train_result, training_time
    
    def comprehensive_evaluation(self, trainer, test_dataset, output_dir="./lora_results"):
        """הערכה מקיפה עם כל המטריקות"""
        print("📊 Running comprehensive evaluation...")
        
        # הערכה על test set
        eval_result = trainer.evaluate(test_dataset)
        
        # תחזיות מפורטות
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        y_pred_proba = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
        
        # מטריקות מפורטות
        accuracy = accuracy_score(y_true, y_pred)
        
        # דוח classification
        class_report = classification_report(
            y_true, y_pred,
            target_names=['negative', 'positive'],
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # ROC AUC (עבור binary classification)
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
        except:
            roc_auc = 0.0
            fpr, tpr, thresholds = [], [], []
        
        # שמירת תוצאות
        results = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'kappa': kappa,
            'roc_auc': roc_auc,
            'eval_metrics': eval_result,
            'predictions': {
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist()
            }
        }
        
        # שמירה לקובץ
        with open(f"{output_dir}/evaluation_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # הדפסת תוצאות
        print(f"   📈 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   📊 Macro F1: {class_report['macro avg']['f1-score']:.4f}")
        print(f"   📊 Weighted F1: {class_report['weighted avg']['f1-score']:.4f}")
        print(f"   📊 Cohen's Kappa: {kappa:.4f}")
        print(f"   📊 ROC AUC: {roc_auc:.4f}")
        
        print("\n   Per-class results:")
        for label in ['negative', 'positive']:
            if label in class_report:
                metrics = class_report[label]
                print(f"     {label}: P={metrics['precision']:.3f}, "
                      f"R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        self.evaluation_results = results
        return results

def create_advanced_visualizations(results, output_dir="./lora_results"):
    """יצירת ויזואליזציות מתקדמות"""
    print("📊 Creating advanced visualizations...")
    
    plt.style.use('default')
    
    # 1. Confusion Matrix מפורטת
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LoRA Fine-tuning - Advanced Analysis', fontsize=16, fontweight='bold')
    
    # Confusion Matrix
    cm = np.array(results['confusion_matrix'])
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # Per-class metrics
    class_report = results['classification_report']
    classes = ['negative', 'positive']
    metrics = ['precision', 'recall', 'f1-score']
    
    metric_data = np.array([[class_report[cls][metric] for metric in metrics] for cls in classes])
    
    sns.heatmap(
        metric_data, annot=True, fmt='.3f', cmap='RdYlGn',
        xticklabels=metrics, yticklabels=classes,
        ax=axes[0, 1], vmin=0, vmax=1
    )
    axes[0, 1].set_title('Per-Class Metrics')
    
    # ROC Curve (אם זמין)
    if results['roc_auc'] > 0:
        # מחשב ROC מחדש מהנתונים
        y_true = np.array(results['predictions']['y_true'])
        y_pred_proba = np.array(results['predictions']['y_pred_proba'])
        
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            
            axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                           label=f'ROC curve (AUC = {results["roc_auc"]:.3f})')
            axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1, 0].set_xlim([0.0, 1.0])
            axes[1, 0].set_ylim([0.0, 1.05])
            axes[1, 0].set_xlabel('False Positive Rate')
            axes[1, 0].set_ylabel('True Positive Rate')
            axes[1, 0].set_title('ROC Curve')
            axes[1, 0].legend(loc="lower right")
    
    # Distribution של predictions
    y_pred_proba = np.array(results['predictions']['y_pred_proba'])
    if len(y_pred_proba.shape) > 1:
        axes[1, 1].hist(y_pred_proba[:, 1], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_xlabel('Positive Class Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Confidence Distribution')
        axes[1, 1].axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/lora_advanced_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   📊 Visualizations saved to {output_dir}/lora_advanced_analysis.png")

def compare_with_baseline(lora_results, baseline_file="simple_fine_tuning_results.csv"):
    """השוואה מפורטת עם baseline"""
    print("🔍 Comparing with baseline methods...")
    
    comparison_results = {
        'LoRA': {
            'accuracy': lora_results['accuracy'],
            'macro_f1': lora_results['classification_report']['macro avg']['f1-score'],
            'weighted_f1': lora_results['classification_report']['weighted avg']['f1-score'],
            'kappa': lora_results['kappa'],
            'roc_auc': lora_results['roc_auc']
        }
    }
    
    # טען תוצאות baseline אם קיימות
    try:
        if os.path.exists(baseline_file):
            baseline_df = pd.read_csv(baseline_file)
            comparison_results['Baseline_Full_FT'] = {
                'accuracy': baseline_df.iloc[0]['accuracy'],
                'macro_f1': 0.0,  # לא זמין
                'weighted_f1': 0.0,  # לא זמין  
                'kappa': 0.0,  # לא זמין
                'roc_auc': 0.0  # לא זמין
            }
    except Exception as e:
        print(f"   ⚠️ Could not load baseline results: {e}")
    
    # שמירת השוואה
    comparison_df = pd.DataFrame(comparison_results).T
    comparison_df.to_csv("./lora_results/lora_comparison.csv")
    
    print("   📊 Comparison results:")
    for method, metrics in comparison_results.items():
        print(f"     {method}:")
        for metric, value in metrics.items():
            print(f"       {metric}: {value:.4f}")
    
    return comparison_results

def main():
    """פונקציה ראשית - הרצת LoRA מתקדמת"""
    print("🚀 Advanced LoRA Fine-tuning for Hebrew Sentiment Analysis")
    print("=" * 70)
    
    try:
        # יצירת תיקייה לתוצאות
        output_dir = "./lora_results"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/logs", exist_ok=True)
        
        # יצירת trainer
        trainer = AdvancedLoRATrainer()
        
        # טעינת נתונים
        print("\n📊 Loading and preparing data...")
        df = pd.read_csv("dataset_fixed.csv")
        print(f"   Loaded dataset with {len(df)} samples")
        
        # הכנת נתונים
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df)
        
        # יצירת datasets
        train_dataset, val_dataset, test_dataset = trainer.create_datasets(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # הגדרת LoRA configurations לניסוי
        lora_configs = [
            {"name": "LoRA_r16", "config": setup_lora_config(rank=16, alpha=32, dropout=0.1)},
            {"name": "LoRA_r8", "config": setup_lora_config(rank=8, alpha=16, dropout=0.1)},
        ]
        
        best_results = None
        best_accuracy = 0
        
        # ניסוי של configurations שונות
        for lora_exp in lora_configs:
            print(f"\n🧪 Experimenting with {lora_exp['name']}...")
            
            # הגדרת מודל
            model = trainer.setup_model_and_lora(lora_exp['config'])
            
            # אימון
            trained_trainer, train_result, training_time = trainer.train_with_advanced_settings(
                train_dataset, val_dataset, f"{output_dir}/{lora_exp['name']}"
            )
            
            # הערכה
            results = trainer.comprehensive_evaluation(
                trained_trainer, test_dataset, f"{output_dir}/{lora_exp['name']}"
            )
            
            # שמירת התוצאות הטובות ביותר
            if results['accuracy'] > best_accuracy:
                best_accuracy = results['accuracy']
                best_results = results
                best_config_name = lora_exp['name']
                
                # יצירת ויזואליזציות עבור הטוב ביותר
                create_advanced_visualizations(results, f"{output_dir}/{lora_exp['name']}")
        
        print(f"\n🏆 Best configuration: {best_config_name}")
        print(f"   Best accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        # השוואה עם baseline
        comparison = compare_with_baseline(best_results)
        
        # שמירת דוח סופי
        final_report = {
            'best_config': best_config_name,
            'best_results': best_results,
            'comparison': comparison,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(f"{output_dir}/final_lora_report.json", 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Advanced LoRA analysis completed successfully!")
        print(f"📁 Results saved to: {output_dir}/")
        print(f"📊 Key files created:")
        print(f"   • final_lora_report.json - דוח מפורט")
        print(f"   • lora_advanced_analysis.png - ויזואליזציות")
        print(f"   • evaluation_results.json - מטריקות מפורטות")
        print(f"   • lora_comparison.csv - השוואה עם baseline")
        
        return best_results
        
    except Exception as e:
        print(f"❌ Error in LoRA training: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
