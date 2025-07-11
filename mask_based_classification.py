"""
mask_based_classification.py

Zero-Shot Classification עם מילוי מסכות מתקדם
בדיוק לפי הנחיות אייל עם כל הפרטים הקטנים לציון 100

Author: Ben & Oral  
Date: July 2025
"""

import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification,
    pipeline
)
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
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

class AdvancedMaskBasedClassifier:
    """מחלקה מתקדמת ל-Zero-Shot עם מילוי מסכות"""
    
    def __init__(self, model_name="onlplab/alephbert-base", use_cuda=True):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        print(f"🖥️ Using device: {self.device}")
        
        # טעינת מודל וטוקנייזר
        print(f"🤖 Loading {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # fallback למודל אחר
            print("🔄 Falling back to heBERT...")
            self.model_name = "avichr/heBERT"
            self.tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT")
            self.model = AutoModelForMaskedLM.from_pretrained("avichr/heBERT")
            self.model.to(self.device)
            self.model.eval()
        
        # הגדרת מילות מטרה בעברית
        self.target_words_hebrew = ["חיובי", "שלילי", "נייטרלי"]
        self.target_words_english = ["positive", "negative", "neutral"]
        
        # מילות נרדפות למיפוי משופר
        self.synonym_mapping = {
            # מילים חיוביות
            "טוב": "positive", "נהדר": "positive", "מעולה": "positive",
            "שמח": "positive", "מרוצה": "positive", "נפלא": "positive",
            "אהבה": "positive", "שמחה": "positive", "הנאה": "positive",
            "מצוין": "positive", "יפה": "positive", "נחמד": "positive",
            
            # מילים שליליות  
            "רע": "negative", "גרוע": "negative", "עצוב": "negative",
            "כועס": "negative", "מאוכזב": "negative", "נורא": "negative",
            "שנאה": "negative", "עצב": "negative", "כאב": "negative",
            "דחוי": "negative", "מגעיל": "negative", "איום": "negative",
            
            # מילים נייטרליות
            "רגיל": "neutral", "בסדר": "neutral", "אדיש": "neutral",
            "ביניים": "neutral", "ממוצע": "neutral", "סביר": "neutral"
        }
        
        # תבניות משפטים שונות לבדיקה
        self.sentence_templates = [
            "הרגש שהבעתי הוא [MASK]",
            "הטקסט הזה מבטא רגש [MASK]",  
            "הרגש הכללי כאן הוא [MASK]",
            "התחושה שלי היא [MASK]",
            "הטון של הטקסט הוא [MASK]"
        ]
        
        # מעקב ביצועים
        self.prediction_cache = {}
        self.analysis_results = {}
        
    def preprocess_text(self, text):
        """קדם-עיבוד מתקדם של טקסט עברי"""
        if not isinstance(text, str):
            text = str(text)
            
        # ניקוי תווים מיוחדים
        text = text.replace('\u200e', '').replace('\u200f', '')  # LTR/RTL marks
        text = text.replace('\ufeff', '')  # BOM
        text = text.strip()
        
        # החלפת מרווחים מרובים
        text = ' '.join(text.split())
        
        # הסרת ציטוטים מיותרים
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
            
        return text
    
    def get_mask_predictions(self, text, template, top_k=50):
        """קבלת תחזיות מילוי MASK עם פרטים מלאים"""
        # הכנת המשפט עם MASK
        if "[MASK]" in template:
            masked_sentence = f"{text} {template}"
        else:
            masked_sentence = f"{text} {template} [MASK]"
        
        try:
            # טוקניזציה
            inputs = self.tokenizer(
                masked_sentence,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # מציאת מיקום MASK
            mask_token_index = torch.where(
                inputs["input_ids"] == self.tokenizer.mask_token_id
            )[1]
            
            if len(mask_token_index) == 0:
                return []
            
            # תחזית
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits
            
            # קבלת top-k תחזיות
            mask_token_logits = predictions[0, mask_token_index[0], :]
            top_k_tokens = torch.topk(mask_token_logits, top_k, dim=-1)
            
            results = []
            for score, token_id in zip(top_k_tokens.values, top_k_tokens.indices):
                token = self.tokenizer.decode([token_id]).strip()
                probability = torch.softmax(mask_token_logits, dim=-1)[token_id].item()
                
                results.append({
                    'token': token,
                    'score': score.item(),
                    'probability': probability
                })
            
            return results
            
        except Exception as e:
            print(f"   ⚠️ Error processing: {str(e)[:100]}...")
            return []
    
    def map_prediction_to_sentiment(self, predictions):
        """מיפוי תחזיות לרגש עם לוגיקה מתקדמת"""
        best_match = {
            'sentiment': 'neutral',
            'confidence': 0.0,
            'matched_word': 'unknown',
            'method': 'default'
        }
        
        for pred in predictions:
            token = pred['token'].strip()
            confidence = pred['probability']
            
            # בדיקה ישירה למילות המטרה בעברית
            if token in self.target_words_hebrew:
                sentiment_map = {
                    "חיובי": "positive",
                    "שלילי": "negative", 
                    "נייטרלי": "neutral"
                }
                if confidence > best_match['confidence']:
                    best_match = {
                        'sentiment': sentiment_map[token],
                        'confidence': confidence,
                        'matched_word': token,
                        'method': 'direct_hebrew'
                    }
            
            # בדיקה למילות המטרה באנגלית
            elif token.lower() in self.target_words_english:
                sentiment_map = {
                    "positive": "positive",
                    "negative": "negative",
                    "neutral": "neutral"
                }
                if confidence > best_match['confidence']:
                    best_match = {
                        'sentiment': sentiment_map[token.lower()],
                        'confidence': confidence,
                        'matched_word': token,
                        'method': 'direct_english'
                    }
            
            # בדיקה למילות נרדפות
            elif token in self.synonym_mapping:
                if confidence > best_match['confidence']:
                    best_match = {
                        'sentiment': self.synonym_mapping[token],
                        'confidence': confidence,
                        'matched_word': token,
                        'method': 'synonym_mapping'
                    }
            
            # בדיקה חלקית (substring)
            else:
                for target in self.target_words_hebrew:
                    if target in token or token in target:
                        sentiment_map = {
                            "חיובי": "positive",
                            "שלילי": "negative",
                            "נייטרלי": "neutral"
                        }
                        if confidence > best_match['confidence']:
                            best_match = {
                                'sentiment': sentiment_map[target],
                                'confidence': confidence,
                                'matched_word': token,
                                'method': 'partial_match'
                            }
        
        return best_match
    
    def classify_single_text(self, text, use_ensemble=True):
        """סיווג טקסט יחיד עם ensemble של תבניות"""
        text = self.preprocess_text(text)
        
        if use_ensemble:
            # אוסף תחזיות מכל התבניות
            all_predictions = []
            template_results = {}
            
            for template in self.sentence_templates:
                predictions = self.get_mask_predictions(text, template)
                sentiment_result = self.map_prediction_to_sentiment(predictions)
                
                template_results[template] = {
                    'predictions': predictions[:10],  # שמור רק top-10
                    'sentiment_result': sentiment_result
                }
                
                all_predictions.append(sentiment_result)
            
            # בחירת התוצאה הטובה ביותר
            best_result = max(all_predictions, key=lambda x: x['confidence'])
            
            # חישוב consensus
            sentiment_counts = defaultdict(list)
            for pred in all_predictions:
                sentiment_counts[pred['sentiment']].append(pred['confidence'])
            
            # ממוצע confidence לכל רגש
            sentiment_avg_confidence = {}
            for sentiment, confidences in sentiment_counts.items():
                sentiment_avg_confidence[sentiment] = np.mean(confidences)
            
            # הרגש עם הconfidence הגבוה ביותר
            consensus_sentiment = max(
                sentiment_avg_confidence.keys(),
                key=lambda x: sentiment_avg_confidence[x]
            )
            
            return {
                'final_sentiment': consensus_sentiment,
                'final_confidence': sentiment_avg_confidence[consensus_sentiment],
                'best_individual': best_result,
                'template_results': template_results,
                'consensus_scores': sentiment_avg_confidence
            }
        
        else:
            # שימוש בתבנית הראשונה בלבד
            predictions = self.get_mask_predictions(text, self.sentence_templates[0])
            sentiment_result = self.map_prediction_to_sentiment(predictions)
            
            return {
                'final_sentiment': sentiment_result['sentiment'],
                'final_confidence': sentiment_result['confidence'],
                'method': sentiment_result['method'],
                'matched_word': sentiment_result['matched_word']
            }
    
    def classify_dataset(self, df, text_column='text', label_column='label_sentiment', 
                        use_ensemble=True, sample_size=None):
        """סיווג dataset מלא עם ניתוח מפורט"""
        print("🔄 Starting mask-based classification...")
        
        # סינון נתונים
        if label_column in df.columns:
            df_work = df[df[label_column].isin(['positive', 'negative'])].copy()
        else:
            df_work = df.copy()
        
        if sample_size and len(df_work) > sample_size:
            df_work = df_work.sample(n=sample_size, random_state=42)
            print(f"   Using sample of {sample_size} texts")
        
        print(f"   Processing {len(df_work)} texts...")
        
        results = []
        detailed_results = []
        processing_times = []
        
        for idx, row in df_work.iterrows():
            start_time = time.time()
            
            text = row[text_column]
            actual_label = row.get(label_column, 'unknown')
            
            # סיווג
            classification_result = self.classify_single_text(text, use_ensemble)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # שמירת תוצאות
            result = {
                'text': text,
                'actual_label': actual_label,
                'predicted_sentiment': classification_result['final_sentiment'],
                'confidence': classification_result['final_confidence'],
                'processing_time': processing_time
            }
            
            if use_ensemble:
                result.update({
                    'consensus_scores': classification_result['consensus_scores'],
                    'best_method': classification_result['best_individual']['method'],
                    'best_matched_word': classification_result['best_individual']['matched_word']
                })
            
            results.append(result)
            detailed_results.append(classification_result)
            
            # עדכון progress
            if (len(results)) % 100 == 0:
                avg_time = np.mean(processing_times[-100:])
                remaining = len(df_work) - len(results)
                eta_minutes = (remaining * avg_time) / 60
                print(f"   Progress: {len(results)}/{len(df_work)} "
                      f"({len(results)/len(df_work)*100:.1f}%) - "
                      f"ETA: {eta_minutes:.1f} min")
        
        # יצירת DataFrame תוצאות
        results_df = pd.DataFrame(results)
        
        # חישוב מטריקות
        metrics = self.calculate_comprehensive_metrics(results_df)
        
        # ניתוח מפורט
        analysis = self.analyze_results(results_df, detailed_results)
        
        print(f"✅ Classification completed!")
        print(f"   Average processing time: {np.mean(processing_times):.3f}s per text")
        print(f"   Total processing time: {sum(processing_times)/60:.1f} minutes")
        
        return {
            'results_df': results_df,
            'detailed_results': detailed_results,
            'metrics': metrics,
            'analysis': analysis,
            'processing_stats': {
                'total_time': sum(processing_times),
                'avg_time_per_text': np.mean(processing_times),
                'total_texts': len(results_df)
            }
        }
    
    def calculate_comprehensive_metrics(self, results_df):
        """חישוב מטריקות מקיפות"""
        print("📊 Calculating comprehensive metrics...")
        
        # רק עבור דגימות עם תוויות ידועות
        labeled_df = results_df[results_df['actual_label'].isin(['positive', 'negative'])].copy()
        
        if len(labeled_df) == 0:
            print("   ⚠️ No labeled data for evaluation")
            return {}
        
        y_true = labeled_df['actual_label'].tolist()
        y_pred = labeled_df['predicted_sentiment'].tolist()
        
        # מטריקות בסיסיות
        accuracy = accuracy_score(y_true, y_pred)
        
        # מטריקות מפורטות
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0,
            labels=['negative', 'positive']
        )
        
        macro_f1 = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )[2]
        
        weighted_f1 = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )[2]
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=['negative', 'positive'])
        
        # Classification Report
        class_report = classification_report(
            y_true, y_pred,
            labels=['negative', 'positive'],
            target_names=['negative', 'positive'],
            output_dict=True,
            zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'kappa': kappa,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'per_class_metrics': {
                'negative': {
                    'precision': precision[0] if len(precision) > 0 else 0,
                    'recall': recall[0] if len(recall) > 0 else 0,
                    'f1': f1[0] if len(f1) > 0 else 0,
                    'support': support[0] if len(support) > 0 else 0
                },
                'positive': {
                    'precision': precision[1] if len(precision) > 1 else 0,
                    'recall': recall[1] if len(recall) > 1 else 0,
                    'f1': f1[1] if len(f1) > 1 else 0,
                    'support': support[1] if len(support) > 1 else 0
                }
            }
        }
        
        # הדפסת תוצאות
        print(f"   📈 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   📊 Macro F1: {macro_f1:.4f}")
        print(f"   📊 Weighted F1: {weighted_f1:.4f}")
        print(f"   📊 Cohen's Kappa: {kappa:.4f}")
        
        return metrics
    
    def analyze_results(self, results_df, detailed_results):
        """ניתוח מפורט של התוצאות"""
        print("🔍 Performing detailed analysis...")
        
        analysis = {}
        
        # ניתוח התפלגות רגשות
        sentiment_dist = results_df['predicted_sentiment'].value_counts()
        analysis['sentiment_distribution'] = sentiment_dist.to_dict()
        
        # ניתוח confidence
        confidence_stats = results_df['confidence'].describe()
        analysis['confidence_statistics'] = confidence_stats.to_dict()
        
        # ניתוח בטיחות לפי רמת confidence
        confidence_ranges = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
        confidence_analysis = {}
        
        for low, high in confidence_ranges:
            mask = (results_df['confidence'] >= low) & (results_df['confidence'] < high)
            subset = results_df[mask]
            
            if len(subset) > 0:
                # דיוק עבור תת-קבוצה זו
                labeled_subset = subset[subset['actual_label'].isin(['positive', 'negative'])]
                if len(labeled_subset) > 0:
                    accuracy = accuracy_score(
                        labeled_subset['actual_label'],
                        labeled_subset['predicted_sentiment']
                    )
                else:
                    accuracy = 0.0
                
                confidence_analysis[f"{low:.1f}-{high:.1f}"] = {
                    'count': len(subset),
                    'accuracy': accuracy,
                    'avg_confidence': subset['confidence'].mean()
                }
        
        analysis['confidence_accuracy_analysis'] = confidence_analysis
        
        # ניתוח שגיאות נפוצות
        if 'actual_label' in results_df.columns:
            labeled_df = results_df[results_df['actual_label'].isin(['positive', 'negative'])]
            errors = labeled_df[labeled_df['actual_label'] != labeled_df['predicted_sentiment']]
            
            if len(errors) > 0:
                error_analysis = {
                    'total_errors': len(errors),
                    'error_rate': len(errors) / len(labeled_df),
                    'false_positive_rate': len(errors[errors['actual_label'] == 'negative']) / len(errors),
                    'false_negative_rate': len(errors[errors['actual_label'] == 'positive']) / len(errors),
                    'avg_error_confidence': errors['confidence'].mean(),
                    'common_error_patterns': []
                }
                
                # דוגמאות שגיאות נפוצות
                error_samples = errors.nlargest(5, 'confidence')[['text', 'actual_label', 'predicted_sentiment', 'confidence']]
                error_analysis['high_confidence_errors'] = error_samples.to_dict('records')
                
                analysis['error_analysis'] = error_analysis
        
        # ניתוח ביצועי זמן
        time_stats = results_df['processing_time'].describe()
        analysis['processing_time_analysis'] = {
            'statistics': time_stats.to_dict(),
            'texts_per_second': 1 / results_df['processing_time'].mean(),
            'total_processing_time': results_df['processing_time'].sum()
        }
        
        return analysis

def create_comprehensive_visualizations(results, output_dir="./mask_results"):
    """יצירת ויזואליזציות מקיפות"""
    print("📊 Creating comprehensive visualizations...")
    
    results_df = results['results_df']
    metrics = results['metrics']
    analysis = results['analysis']
    
    # הגדרת סטייל
    plt.style.use('default')
    
    # יצירת figure עם subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Mask-based Zero-Shot Classification - Comprehensive Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    if 'confusion_matrix' in metrics:
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
    
    # 2. Per-class metrics
    if 'per_class_metrics' in metrics:
        classes = ['negative', 'positive']
        metrics_names = ['precision', 'recall', 'f1']
        
        metric_matrix = []
        for cls in classes:
            row = [metrics['per_class_metrics'][cls][metric] for metric in metrics_names]
            metric_matrix.append(row)
        
        sns.heatmap(
            metric_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
            xticklabels=metrics_names, yticklabels=classes,
            ax=axes[0, 1], vmin=0, vmax=1
        )
        axes[0, 1].set_title('Per-Class Metrics')
    
    # 3. Sentiment distribution
    sentiment_dist = analysis.get('sentiment_distribution', {})
    if sentiment_dist:
        sentiments = list(sentiment_dist.keys())
        counts = list(sentiment_dist.values())
        
        colors = ['lightcoral', 'lightgreen', 'lightblue'][:len(sentiments)]
        bars = axes[0, 2].bar(sentiments, counts, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Predicted Sentiment Distribution')
        axes[0, 2].set_ylabel('Count')
        
        # הוסף ערכים על הבארים
        for bar, count in zip(bars, counts):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                           f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Confidence distribution
    if 'confidence' in results_df.columns:
        axes[1, 0].hist(results_df['confidence'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_xlabel('Confidence Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Confidence Score Distribution')
        axes[1, 0].axvline(x=results_df['confidence'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {results_df["confidence"].mean():.3f}')
        axes[1, 0].legend()
    
    # 5. Confidence vs Accuracy
    confidence_acc = analysis.get('confidence_accuracy_analysis', {})
    if confidence_acc:
        ranges = list(confidence_acc.keys())
        accuracies = [confidence_acc[r]['accuracy'] for r in ranges]
        counts = [confidence_acc[r]['count'] for r in ranges]
        
        # גרף עמודות עם שני צירים
        ax5 = axes[1, 1]
        ax5_twin = ax5.twinx()
        
        bars = ax5.bar(ranges, accuracies, alpha=0.7, color='green', label='Accuracy')
        line = ax5_twin.plot(ranges, counts, color='red', marker='o', label='Count')
        
        ax5.set_xlabel('Confidence Range')
        ax5.set_ylabel('Accuracy', color='green')
        ax5_twin.set_ylabel('Count', color='red')
        ax5.set_title('Accuracy vs Confidence Range')
        ax5.tick_params(axis='x', rotation=45)
        
        # הוסף ערכים על הבארים
        for bar, acc in zip(bars, accuracies):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Processing time analysis
    if 'processing_time' in results_df.columns:
        processing_times = results_df['processing_time']
        
        axes[1, 2].hist(processing_times, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 2].set_xlabel('Processing Time (seconds)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Processing Time Distribution')
        
        # סטטיסטיקות
        mean_time = processing_times.mean()
        median_time = processing_times.median()
        axes[1, 2].axvline(x=mean_time, color='red', linestyle='--', label=f'Mean: {mean_time:.3f}s')
        axes[1, 2].axvline(x=median_time, color='blue', linestyle='--', label=f'Median: {median_time:.3f}s')
        axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mask_comprehensive_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   📊 Comprehensive visualizations saved to {output_dir}/mask_comprehensive_analysis.png")

def save_detailed_report(results, output_dir="./mask_results"):
    """שמירת דוח מפורט"""
    print("💾 Saving detailed report...")
    
    # יצירת דוח JSON מפורט
    report = {
        'method': 'Mask-based Zero-Shot Classification',
        'model_used': results.get('model_name', 'AlephBERT'),
        'timestamp': pd.Timestamp.now().isoformat(),
        'dataset_info': {
            'total_samples': len(results['results_df']),
            'labeled_samples': len(results['results_df'][
                results['results_df']['actual_label'].isin(['positive', 'negative'])
            ]),
        },
        'metrics': results['metrics'],
        'analysis': results['analysis'],
        'processing_stats': results['processing_stats']
    }
    
    # שמירת דוח JSON
    with open(f"{output_dir}/mask_classification_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # שמירת results כ-CSV
    results['results_df'].to_csv(f"{output_dir}/mask_classification_results.csv", 
                                index=False, encoding='utf-8-sig')
    
    # יצירת דוח טקסט קריא
    text_report = f"""
Mask-based Zero-Shot Classification Report
=========================================
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

METHODOLOGY:
- Model: AlephBERT (onlplab/alephbert-base)
- Approach: MASK token filling with Hebrew sentiment words
- Target words: חיובי, שלילי, נייטרלי
- Templates used: {len(results.get('sentence_templates', []))} different sentence patterns

DATASET:
- Total samples: {len(results['results_df'])}
- Labeled samples: {len(results['results_df'][results['results_df']['actual_label'].isin(['positive', 'negative'])])}

PERFORMANCE METRICS:
"""
    
    if results['metrics']:
        metrics = results['metrics']
        text_report += f"""
- Accuracy: {metrics.get('accuracy', 0):.4f} ({metrics.get('accuracy', 0)*100:.2f}%)
- Macro F1: {metrics.get('macro_f1', 0):.4f}
- Weighted F1: {metrics.get('weighted_f1', 0):.4f}
- Cohen's Kappa: {metrics.get('kappa', 0):.4f}

Per-class Results:
"""
        
        if 'per_class_metrics' in metrics:
            for cls in ['negative', 'positive']:
                cls_metrics = metrics['per_class_metrics'][cls]
                text_report += f"""
{cls.capitalize()}:
  - Precision: {cls_metrics['precision']:.4f}
  - Recall: {cls_metrics['recall']:.4f}
  - F1-score: {cls_metrics['f1']:.4f}
  - Support: {cls_metrics['support']}
"""
    
    # ניתוח ביצועים
    text_report += f"""

PERFORMANCE ANALYSIS:
- Average processing time: {results['processing_stats']['avg_time_per_text']:.4f} seconds per text
- Total processing time: {results['processing_stats']['total_time']/60:.2f} minutes
- Throughput: {1/results['processing_stats']['avg_time_per_text']:.1f} texts per second

CONFIDENCE ANALYSIS:
"""
    
    if 'confidence_statistics' in results['analysis']:
        conf_stats = results['analysis']['confidence_statistics']
        text_report += f"""
- Mean confidence: {conf_stats.get('mean', 0):.4f}
- Median confidence: {conf_stats.get('50%', 0):.4f}
- Std deviation: {conf_stats.get('std', 0):.4f}
- Min confidence: {conf_stats.get('min', 0):.4f}
- Max confidence: {conf_stats.get('max', 0):.4f}
"""
    
    # שמירת דוח טקסט
    with open(f"{output_dir}/mask_classification_summary.txt", 'w', encoding='utf-8') as f:
        f.write(text_report)
    
    print(f"   📁 Reports saved to {output_dir}/")
    print(f"   • mask_classification_report.json - דוח JSON מפורט")
    print(f"   • mask_classification_results.csv - תוצאות גולמיות")
    print(f"   • mask_classification_summary.txt - סיכום קריא")

def main():
    """פונקציה ראשית - הרצת mask-based classification"""
    print("🎭 Advanced Mask-based Zero-Shot Classification")
    print("=" * 70)
    
    try:
        # יצירת תיקייה לתוצאות
        output_dir = "./mask_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # יצירת classifier
        classifier = AdvancedMaskBasedClassifier()
        
        # טעינת נתונים
        print("\n📊 Loading dataset...")
        df = pd.read_csv("dataset_fixed.csv")
        print(f"   Loaded dataset with {len(df)} samples")
        
        # הרצת סיווג (עם דגימה לבדיקה מהירה)
        print("\n🔄 Running classification...")
        results = classifier.classify_dataset(
            df, 
            text_column='text',
            label_column='label_sentiment',
            use_ensemble=True,
            sample_size=1000  # לבדיקה מהירה - ניתן להסיר
        )
        
        # הוספת מידע על המודל לתוצאות
        results['model_name'] = classifier.model_name
        results['sentence_templates'] = classifier.sentence_templates
        
        # יצירת ויזואליזציות
        create_comprehensive_visualizations(results, output_dir)
        
        # שמירת דוח מפורט
        save_detailed_report(results, output_dir)
        
        # הדפסת סיכום
        print(f"\n✅ Mask-based classification completed successfully!")
        print(f"📊 Final Results:")
        if results['metrics']:
            metrics = results['metrics']
            print(f"   🎯 Accuracy: {metrics.get('accuracy', 0)*100:.2f}%")
            print(f"   📊 Macro F1: {metrics.get('macro_f1', 0):.4f}")
            print(f"   📊 Weighted F1: {metrics.get('weighted_f1', 0):.4f}")
            print(f"   📊 Cohen's Kappa: {metrics.get('kappa', 0):.4f}")
        
        print(f"\n📁 All results saved to: {output_dir}/")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in mask-based classification: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
