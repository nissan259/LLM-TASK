"""
advanced_metrics_analysis.py

מטריקות מתקדמות וניתוח סטטיסטי מעמיק
בדיוק לפי הנחיות אייל עם כל הפרטים הקטנים לציון 100

Author: Ben & Oral  
Date: July 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, cohen_kappa_score, matthews_corrcoef,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    balanced_accuracy_score, f1_score, jaccard_score, log_loss
)
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.model_selection import permutation_test_score, cross_val_score
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, binomtest
import json
import os
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

class AdvancedMetricsAnalyzer:
    """מחלקה לניתוח מטריקות מתקדם וסטטיסטיקה מעמיקה"""
    
    def __init__(self, output_dir="./advanced_metrics"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # הגדרת פונטים לעברית
        plt.rcParams['font.family'] = ['Arial Unicode MS', 'Tahoma', 'DejaVu Sans']
        
        # שמירת תוצאות לניתוח השוואתי
        self.all_results = {}
        
    def load_all_results(self):
        """טעינת כל התוצאות מהקבצים השונים"""
        print("📂 Loading all experimental results...")
        
        results_files = {
            'simple_fine_tuning': './simple_fine_tuning_results.csv',
            'simple_peft': './simple_peft_results.csv', 
            'zero_shot_bart': './zero_shot_bart_summary.csv',
            'zero_shot_mask': './mask_zero_shot_summary.csv',
            'lora_advanced': './lora_results.csv',  # יווצר על ידי הscript הקודם
            'mask_advanced': './mask_results/mask_classification_results.csv'
        }
        
        for method, file_path in results_files.items():
            try:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    self.all_results[method] = df
                    print(f"   ✅ Loaded {method}: {len(df)} samples")
                else:
                    print(f"   ⚠️ File not found: {file_path}")
            except Exception as e:
                print(f"   ❌ Error loading {method}: {e}")
        
        print(f"📊 Total methods loaded: {len(self.all_results)}")
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_proba=None, method_name="Unknown"):
        """חישוב מטריקות מקיפות לכל שיטה"""
        print(f"📊 Calculating comprehensive metrics for {method_name}...")
        
        # בדיקת תקינות נתונים
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # המרה לבינארי אם נדרש
        if set(y_true) == {'positive', 'negative'}:
            y_true_binary = [1 if label == 'positive' else 0 for label in y_true]
            y_pred_binary = [1 if label == 'positive' else 0 for label in y_pred]
        else:
            y_true_binary = y_true
            y_pred_binary = y_pred
        
        metrics = {}
        
        # מטריקות בסיסיות
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics['precision_negative'] = precision[0] if len(precision) > 0 else 0
        metrics['precision_positive'] = precision[1] if len(precision) > 1 else 0
        metrics['recall_negative'] = recall[0] if len(recall) > 0 else 0
        metrics['recall_positive'] = recall[1] if len(recall) > 1 else 0
        metrics['f1_negative'] = f1[0] if len(f1) > 0 else 0
        metrics['f1_positive'] = f1[1] if len(f1) > 1 else 0
        
        # מטריקות מקרו ומשוקללות
        metrics['macro_precision'] = precision_recall_fscore_support(y_true, y_pred, average='macro')[0]
        metrics['macro_recall'] = precision_recall_fscore_support(y_true, y_pred, average='macro')[1]
        metrics['macro_f1'] = precision_recall_fscore_support(y_true, y_pred, average='macro')[2]
        
        metrics['weighted_precision'] = precision_recall_fscore_support(y_true, y_pred, average='weighted')[0]
        metrics['weighted_recall'] = precision_recall_fscore_support(y_true, y_pred, average='weighted')[1]
        metrics['weighted_f1'] = precision_recall_fscore_support(y_true, y_pred, average='weighted')[2]
        
        # מטריקות מתקדמות
        metrics['cohens_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true_binary, y_pred_binary)
        metrics['jaccard_score'] = jaccard_score(y_true_binary, y_pred_binary, average='macro')
        
        # Confusion Matrix וחישובים נגזרים
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            # מטריקות נוספות מהconfusion matrix
            metrics['true_negative'] = int(tn)
            metrics['false_positive'] = int(fp)
            metrics['false_negative'] = int(fn)
            metrics['true_positive'] = int(tp)
            
            # שיעור מצבים שונים
            total = tn + fp + fn + tp
            metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
            metrics['false_positive_rate'] = fp / (tn + fp) if (tn + fp) > 0 else 0
            metrics['false_negative_rate'] = fn / (tp + fn) if (tp + fn) > 0 else 0
            metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
            
            # Positive/Negative Predictive Values
            metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
            metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # Likelihood Ratios
            tpr = metrics['true_positive_rate']
            fpr = metrics['false_positive_rate']
            tnr = metrics['true_negative_rate']
            fnr = metrics['false_negative_rate']
            
            metrics['positive_likelihood_ratio'] = tpr / fpr if fpr > 0 else float('inf')
            metrics['negative_likelihood_ratio'] = fnr / tnr if tnr > 0 else float('inf')
            
            # Diagnostic Odds Ratio
            if fp > 0 and fn > 0:
                metrics['diagnostic_odds_ratio'] = (tp * tn) / (fp * fn)
            else:
                metrics['diagnostic_odds_ratio'] = float('inf')
        
        # מטריקות המבוססות על הסתברויות
        if y_proba is not None:
            try:
                # בדיקה אם y_proba הוא ווקטור או מטריצה
                if len(y_proba.shape) == 1:
                    y_proba_positive = y_proba
                else:
                    # אם זה מטריצה, קח את העמודה השנייה (positive)
                    y_proba_positive = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
                
                # ROC AUC
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true_binary, y_proba_positive)
                except:
                    metrics['roc_auc'] = None
                
                # Average Precision (PR AUC)
                try:
                    metrics['average_precision'] = average_precision_score(y_true_binary, y_proba_positive)
                except:
                    metrics['average_precision'] = None
                
                # Log Loss
                try:
                    # ודא שההסתברויות תקינות
                    y_proba_clipped = np.clip(y_proba_positive, 1e-15, 1 - 1e-15)
                    y_proba_2class = np.column_stack([1 - y_proba_clipped, y_proba_clipped])
                    metrics['log_loss'] = log_loss(y_true_binary, y_proba_2class)
                except:
                    metrics['log_loss'] = None
                
                # Brier Score
                try:
                    brier_score = np.mean((y_proba_positive - y_true_binary) ** 2)
                    metrics['brier_score'] = brier_score
                except:
                    metrics['brier_score'] = None
                    
            except Exception as e:
                print(f"   ⚠️ Warning: Could not calculate probability-based metrics: {e}")
                metrics['roc_auc'] = None
                metrics['average_precision'] = None
                metrics['log_loss'] = None
                metrics['brier_score'] = None
        
        # Class distribution
        class_counts = pd.Series(y_true).value_counts()
        metrics['class_distribution'] = class_counts.to_dict()
        
        # Prediction distribution
        pred_counts = pd.Series(y_pred).value_counts()
        metrics['prediction_distribution'] = pred_counts.to_dict()
        
        print(f"   ✅ Calculated {len([k for k, v in metrics.items() if v is not None])} metrics")
        return metrics
    
    def statistical_significance_testing(self, results_dict):
        """בדיקות מובהקות סטטיסטית בין שיטות"""
        print("🔬 Performing statistical significance testing...")
        
        significance_results = {}
        
        # רק שיטות עם תוצאות
        available_methods = list(results_dict.keys())
        
        if len(available_methods) < 2:
            print("   ⚠️ Need at least 2 methods for comparison")
            return significance_results
        
        # השוואה זוגית של כל השיטות
        for i, method1 in enumerate(available_methods):
            for j, method2 in enumerate(available_methods[i+1:], i+1):
                print(f"   🔍 Comparing {method1} vs {method2}")
                
                try:
                    # קבל נתונים משותפים
                    data1 = results_dict[method1]
                    data2 = results_dict[method2]
                    
                    # מצא דגימות משותפות (אם יש עמודת ID או text)
                    if 'text' in data1.columns and 'text' in data2.columns:
                        # מיזוג לפי טקסט
                        merged = data1.merge(data2, on='text', suffixes=('_1', '_2'), how='inner')
                        
                        if len(merged) > 10:  # מספר מינימלי לבדיקה
                            # בדיקת McNemar עבור accuracy
                            correct1 = (merged['actual_label_1'] == merged['predicted_sentiment_1']).astype(int)
                            correct2 = (merged['actual_label_2'] == merged['predicted_sentiment_2']).astype(int)
                            
                            # יצירת טבלת contingency
                            contingency_table = pd.crosstab(correct1, correct2)
                            
                            if contingency_table.shape == (2, 2):
                                # McNemar test
                                b = contingency_table.iloc[0, 1]  # שיטה 1 נכונה, שיטה 2 שגויה
                                c = contingency_table.iloc[1, 0]  # שיטה 1 שגויה, שיטה 2 נכונה
                                
                                if b + c > 0:
                                    mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
                                    mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, 1)
                                else:
                                    mcnemar_stat = 0
                                    mcnemar_p = 1.0
                                
                                # Paired t-test על confidence scores (אם קיימים)
                                ttest_stat, ttest_p = None, None
                                if 'confidence_1' in merged.columns and 'confidence_2' in merged.columns:
                                    try:
                                        ttest_stat, ttest_p = stats.ttest_rel(
                                            merged['confidence_1'], merged['confidence_2']
                                        )
                                    except:
                                        pass
                                
                                # Wilcoxon signed-rank test
                                wilcoxon_stat, wilcoxon_p = None, None
                                try:
                                    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(correct1, correct2)
                                except:
                                    pass
                                
                                # Effect size (Cohen's d)
                                cohens_d = None
                                if 'confidence_1' in merged.columns and 'confidence_2' in merged.columns:
                                    try:
                                        diff = merged['confidence_1'] - merged['confidence_2']
                                        cohens_d = diff.mean() / diff.std()
                                    except:
                                        pass
                                
                                significance_results[f"{method1}_vs_{method2}"] = {
                                    'n_samples': len(merged),
                                    'method1_accuracy': correct1.mean(),
                                    'method2_accuracy': correct2.mean(),
                                    'accuracy_difference': correct1.mean() - correct2.mean(),
                                    'mcnemar_statistic': mcnemar_stat,
                                    'mcnemar_p_value': mcnemar_p,
                                    'mcnemar_significant': mcnemar_p < 0.05,
                                    'ttest_statistic': ttest_stat,
                                    'ttest_p_value': ttest_p,
                                    'ttest_significant': ttest_p < 0.05 if ttest_p else None,
                                    'wilcoxon_statistic': wilcoxon_stat,
                                    'wilcoxon_p_value': wilcoxon_p,
                                    'wilcoxon_significant': wilcoxon_p < 0.05 if wilcoxon_p else None,
                                    'cohens_d': cohens_d,
                                    'effect_size': self.interpret_cohens_d(cohens_d) if cohens_d else None,
                                    'contingency_table': contingency_table.to_dict()
                                }
                                
                                print(f"     📊 McNemar p-value: {mcnemar_p:.4f} "
                                      f"({'Significant' if mcnemar_p < 0.05 else 'Not significant'})")
                            
                        else:
                            print(f"     ⚠️ Insufficient overlapping samples: {len(merged)}")
                    
                except Exception as e:
                    print(f"     ❌ Error comparing {method1} vs {method2}: {e}")
        
        return significance_results
    
    def interpret_cohens_d(self, d):
        """פירוש Cohen's d effect size"""
        if d is None:
            return None
        
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Very small"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def bootstrap_confidence_intervals(self, y_true, y_pred, metric_func, n_bootstrap=1000, confidence_level=0.95):
        """חישוב רווחי בטחון באמצעות bootstrap"""
        print(f"🔄 Computing bootstrap confidence intervals (n={n_bootstrap})...")
        
        n_samples = len(y_true)
        bootstrap_scores = []
        
        for i in range(n_bootstrap):
            # דגימת bootstrap
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = [y_true[idx] for idx in indices]
            y_pred_boot = [y_pred[idx] for idx in indices]
            
            # חישוב מטריקה
            try:
                score = metric_func(y_true_boot, y_pred_boot)
                bootstrap_scores.append(score)
            except:
                continue
        
        if len(bootstrap_scores) == 0:
            return None, None, None
        
        # חישוב רווח בטחון
        alpha = 1 - confidence_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        ci_lower = np.percentile(bootstrap_scores, lower_percentile)
        ci_upper = np.percentile(bootstrap_scores, upper_percentile)
        mean_score = np.mean(bootstrap_scores)
        
        return mean_score, ci_lower, ci_upper
    
    def advanced_error_analysis(self, y_true, y_pred, texts=None, method_name="Unknown"):
        """ניתוח שגיאות מתקדם"""
        print(f"🔍 Performing advanced error analysis for {method_name}...")
        
        analysis = {}
        
        # מציאת שגיאות
        errors_mask = np.array(y_true) != np.array(y_pred)
        errors_indices = np.where(errors_mask)[0]
        
        analysis['total_errors'] = len(errors_indices)
        analysis['error_rate'] = len(errors_indices) / len(y_true)
        
        if len(errors_indices) > 0:
            # סוגי שגיאות
            error_types = defaultdict(int)
            for idx in errors_indices:
                error_type = f"{y_true[idx]}_predicted_as_{y_pred[idx]}"
                error_types[error_type] += 1
            
            analysis['error_types'] = dict(error_types)
            
            # ניתוח אורך טקסט (אם טקסטים זמינים)
            if texts is not None:
                texts = np.array(texts)
                
                # אורכי טקסטים שגויים לעומת נכונים
                error_texts = texts[errors_mask]
                correct_texts = texts[~errors_mask]
                
                error_lengths = [len(str(text).split()) for text in error_texts]
                correct_lengths = [len(str(text).split()) for text in correct_texts]
                
                analysis['error_text_lengths'] = {
                    'mean': np.mean(error_lengths),
                    'median': np.median(error_lengths),
                    'std': np.std(error_lengths)
                }
                
                analysis['correct_text_lengths'] = {
                    'mean': np.mean(correct_lengths),
                    'median': np.median(correct_lengths),
                    'std': np.std(correct_lengths)
                }
                
                # בדיקת הבדל סטטיסטי באורכים
                if len(error_lengths) > 1 and len(correct_lengths) > 1:
                    try:
                        tstat, pval = stats.ttest_ind(error_lengths, correct_lengths)
                        analysis['length_difference_test'] = {
                            't_statistic': tstat,
                            'p_value': pval,
                            'significant': pval < 0.05
                        }
                    except:
                        analysis['length_difference_test'] = None
        
        return analysis
    
    def create_comprehensive_comparison_plots(self, all_metrics):
        """יצירת גרפים השוואתיים מקיפים"""
        print("📊 Creating comprehensive comparison plots...")
        
        # הכנת נתונים לגרפים
        methods = list(all_metrics.keys())
        
        # מטריקות עיקריות להשוואה
        main_metrics = ['accuracy', 'macro_f1', 'weighted_f1', 'cohens_kappa', 'matthews_corrcoef']
        
        # יצירת DataFrame למטריקות
        metrics_df = pd.DataFrame()
        for method in methods:
            method_metrics = {}
            for metric in main_metrics:
                value = all_metrics[method].get(metric)
                if value is not None and not np.isnan(value):
                    method_metrics[metric] = value
                else:
                    method_metrics[metric] = 0
            
            method_metrics['method'] = method
            metrics_df = pd.concat([metrics_df, pd.DataFrame([method_metrics])], ignore_index=True)
        
        # יצירת figure עם subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comprehensive Methods Comparison - Advanced Metrics Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 1. השוואת מטריקות עיקריות
        metrics_for_plot = metrics_df.set_index('method')[main_metrics]
        metrics_for_plot.plot(kind='bar', ax=axes[0, 0], width=0.8)
        axes[0, 0].set_title('Main Performance Metrics Comparison')
        axes[0, 0].set_xlabel('Methods')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Radar Chart למטריקות מנורמלות
        from math import pi
        
        # נרמול מטריקות ל-[0,1]
        normalized_metrics = metrics_for_plot.copy()
        for col in normalized_metrics.columns:
            col_max = normalized_metrics[col].max()
            col_min = normalized_metrics[col].min()
            if col_max != col_min:
                normalized_metrics[col] = (normalized_metrics[col] - col_min) / (col_max - col_min)
        
        # Radar chart
        ax_radar = axes[0, 1]
        angles = [n / float(len(main_metrics)) * 2 * pi for n in range(len(main_metrics))]
        angles += angles[:1]
        
        ax_radar.set_theta_offset(pi / 2)
        ax_radar.set_theta_direction(-1)
        ax_radar.set_thetagrids(np.degrees(angles[:-1]), main_metrics)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
        for i, method in enumerate(methods):
            if method in normalized_metrics.index:
                values = normalized_metrics.loc[method].tolist()
                values += values[:1]
                ax_radar.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
                ax_radar.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Normalized Metrics Radar Chart')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        
        # 3. Per-class Performance
        precision_data = []
        recall_data = []
        f1_data = []
        
        for method in methods:
            metrics = all_metrics[method]
            precision_data.append([
                metrics.get('precision_negative', 0),
                metrics.get('precision_positive', 0)
            ])
            recall_data.append([
                metrics.get('recall_negative', 0),
                metrics.get('recall_positive', 0)
            ])
            f1_data.append([
                metrics.get('f1_negative', 0),
                metrics.get('f1_positive', 0)
            ])
        
        # Grouped bar chart לF1
        x = np.arange(len(methods))
        width = 0.35
        
        f1_neg = [f1[0] for f1 in f1_data]
        f1_pos = [f1[1] for f1 in f1_data]
        
        axes[0, 2].bar(x - width/2, f1_neg, width, label='Negative F1', alpha=0.7, color='lightcoral')
        axes[0, 2].bar(x + width/2, f1_pos, width, label='Positive F1', alpha=0.7, color='lightgreen')
        
        axes[0, 2].set_xlabel('Methods')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].set_title('Per-Class F1 Scores')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(methods, rotation=45)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. ROC Curves (אם יש נתוני הסתברות)
        axes[1, 0].set_title('ROC AUC Comparison')
        roc_aucs = []
        for method in methods:
            roc_auc = all_metrics[method].get('roc_auc')
            if roc_auc is not None:
                roc_aucs.append(roc_auc)
            else:
                roc_aucs.append(0)
        
        bars = axes[1, 0].bar(methods, roc_aucs, alpha=0.7, color=colors)
        axes[1, 0].set_xlabel('Methods')
        axes[1, 0].set_ylabel('ROC AUC')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # הוסף ערכים על הבארים
        for bar, auc in zip(bars, roc_aucs):
            if auc > 0:
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Confidence Distribution (אם יש)
        axes[1, 1].set_title('Method Performance Distribution')
        performance_data = []
        for method in methods:
            performance_data.append([
                all_metrics[method].get('accuracy', 0),
                all_metrics[method].get('macro_f1', 0),
                all_metrics[method].get('cohens_kappa', 0)
            ])
        
        performance_df = pd.DataFrame(performance_data, 
                                    columns=['Accuracy', 'Macro F1', 'Cohen\'s Kappa'],
                                    index=methods)
        
        sns.heatmap(performance_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=axes[1, 1], cbar_kws={'label': 'Score'})
        axes[1, 1].set_title('Performance Heatmap')
        
        # 6. Statistical Significance Matrix
        axes[1, 2].set_title('Method Ranking')
        
        # חישוב ציון מצטבר
        composite_scores = []
        for method in methods:
            metrics = all_metrics[method]
            # ציון משוקלל של מטריקות עיקריות
            score = (
                metrics.get('accuracy', 0) * 0.3 +
                metrics.get('macro_f1', 0) * 0.25 +
                metrics.get('cohens_kappa', 0) * 0.2 +
                metrics.get('matthews_corrcoef', 0) * 0.15 +
                metrics.get('roc_auc', 0) * 0.1 if metrics.get('roc_auc') else 0
            )
            composite_scores.append(score)
        
        # מיון ויצירת ranking
        ranking_df = pd.DataFrame({
            'Method': methods,
            'Composite Score': composite_scores
        }).sort_values('Composite Score', ascending=False)
        
        bars = axes[1, 2].barh(range(len(ranking_df)), ranking_df['Composite Score'], 
                              color=plt.cm.RdYlGn(ranking_df['Composite Score']))
        axes[1, 2].set_yticks(range(len(ranking_df)))
        axes[1, 2].set_yticklabels(ranking_df['Method'])
        axes[1, 2].set_xlabel('Composite Performance Score')
        axes[1, 2].set_title('Method Ranking by Composite Score')
        
        # הוסף ערכים על הבארים
        for i, (bar, score) in enumerate(zip(bars, ranking_df['Composite Score'])):
            axes[1, 2].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                           f'{score:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/comprehensive_methods_comparison_detailed.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   📊 Comprehensive comparison plot saved to {self.output_dir}/")
        
        return ranking_df
    
    def generate_statistical_report(self, all_metrics, significance_results, ranking_df):
        """יצירת דוח סטטיסטי מפורט"""
        print("📝 Generating comprehensive statistical report...")
        
        report = {
            'analysis_metadata': {
                'generated_at': pd.Timestamp.now().isoformat(),
                'methods_analyzed': list(all_metrics.keys()),
                'total_comparisons': len(significance_results)
            },
            'method_rankings': ranking_df.to_dict('records'),
            'detailed_metrics': all_metrics,
            'statistical_significance': significance_results
        }
        
        # שמירת דוח JSON
        with open(f"{self.output_dir}/comprehensive_statistical_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        # יצירת דוח טקסט
        text_report = f"""
Advanced Statistical Analysis Report
==================================
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY:
"""
        
        # טופ 3 שיטות
        top_methods = ranking_df.head(3)
        text_report += f"""
Top 3 Performing Methods:
1. {top_methods.iloc[0]['Method']}: {top_methods.iloc[0]['Composite Score']:.4f}
2. {top_methods.iloc[1]['Method']}: {top_methods.iloc[1]['Composite Score']:.4f}
3. {top_methods.iloc[2]['Method']}: {top_methods.iloc[2]['Composite Score']:.4f}

DETAILED METHOD ANALYSIS:
"""
        
        # פירוט לכל שיטה
        for method in all_metrics.keys():
            metrics = all_metrics[method]
            text_report += f"""
{method.upper()}:
  Core Metrics:
    - Accuracy: {metrics.get('accuracy', 0):.4f} ({metrics.get('accuracy', 0)*100:.2f}%)
    - Macro F1: {metrics.get('macro_f1', 0):.4f}
    - Weighted F1: {metrics.get('weighted_f1', 0):.4f}
    - Cohen's Kappa: {metrics.get('cohens_kappa', 0):.4f}
    - Matthews Correlation: {metrics.get('matthews_corrcoef', 0):.4f}
  
  Per-Class Performance:
    Negative Class:
      - Precision: {metrics.get('precision_negative', 0):.4f}
      - Recall: {metrics.get('recall_negative', 0):.4f}
      - F1-Score: {metrics.get('f1_negative', 0):.4f}
    
    Positive Class:
      - Precision: {metrics.get('precision_positive', 0):.4f}
      - Recall: {metrics.get('recall_positive', 0):.4f}
      - F1-Score: {metrics.get('f1_positive', 0):.4f}
  
  Advanced Metrics:
    - ROC AUC: {metrics.get('roc_auc', 'N/A')}
    - Average Precision: {metrics.get('average_precision', 'N/A')}
    - Brier Score: {metrics.get('brier_score', 'N/A')}
    - Jaccard Score: {metrics.get('jaccard_score', 0):.4f}
"""
        
        # ניתוח מובהקות
        if significance_results:
            text_report += f"""

STATISTICAL SIGNIFICANCE ANALYSIS:
"""
            
            significant_differences = []
            for comparison, results in significance_results.items():
                if results.get('mcnemar_significant', False):
                    significant_differences.append({
                        'comparison': comparison,
                        'p_value': results['mcnemar_p_value'],
                        'accuracy_diff': results['accuracy_difference']
                    })
            
            if significant_differences:
                text_report += f"""
Statistically Significant Differences Found ({len(significant_differences)} comparisons):
"""
                for diff in significant_differences:
                    text_report += f"""
  • {diff['comparison']}: p={diff['p_value']:.4f}, Δacc={diff['accuracy_diff']:.4f}
"""
            else:
                text_report += """
No statistically significant differences found between methods.
"""
        
        # המלצות
        text_report += f"""

RECOMMENDATIONS:
"""
        
        best_method = ranking_df.iloc[0]['Method']
        best_score = ranking_df.iloc[0]['Composite Score']
        
        text_report += f"""
1. BEST OVERALL METHOD: {best_method}
   - Composite Score: {best_score:.4f}
   - Recommended for production use
   
2. PERFORMANCE INSIGHTS:
   - Best Accuracy: {max(all_metrics.values(), key=lambda x: x.get('accuracy', 0))['accuracy']:.4f}
   - Best F1 (Macro): {max(all_metrics.values(), key=lambda x: x.get('macro_f1', 0))['macro_f1']:.4f}
   - Most Balanced: Based on Cohen's Kappa scores
   
3. CONSIDERATIONS:
   - Training time vs performance trade-off
   - Model size and deployment constraints
   - Specific use-case requirements (precision vs recall)
"""
        
        # שמירת דוח טקסט
        with open(f"{self.output_dir}/statistical_analysis_report.txt", 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        print(f"   📄 Statistical reports saved to {self.output_dir}/")
        print(f"   • comprehensive_statistical_analysis.json - דוח JSON מפורט")
        print(f"   • statistical_analysis_report.txt - סיכום קריא")

def main():
    """פונקציה ראשית להרצת ניתוח מטריקות מתקדם"""
    print("🔬 Advanced Metrics & Statistical Analysis")
    print("=" * 70)
    
    try:
        # יצירת analyzer
        analyzer = AdvancedMetricsAnalyzer()
        
        # טעינת כל התוצאות
        analyzer.load_all_results()
        
        if not analyzer.all_results:
            print("❌ No results found to analyze!")
            return
        
        # חישוב מטריקות מקיפות לכל שיטה
        all_metrics = {}
        
        for method_name, df in analyzer.all_results.items():
            print(f"\n📊 Analyzing {method_name}...")
            
            # ודא שיש עמודות נדרשות
            if 'actual_label' in df.columns and 'predicted_sentiment' in df.columns:
                # סינון לתוויות קיימות
                filtered_df = df[df['actual_label'].isin(['positive', 'negative'])].copy()
                
                if len(filtered_df) > 0:
                    y_true = filtered_df['actual_label'].tolist()
                    y_pred = filtered_df['predicted_sentiment'].tolist()
                    
                    # הסתברויות אם קיימות
                    y_proba = None
                    if 'confidence' in filtered_df.columns:
                        y_proba = filtered_df['confidence'].values
                    
                    # חישוב מטריקות
                    metrics = analyzer.calculate_comprehensive_metrics(
                        y_true, y_pred, y_proba, method_name
                    )
                    
                    # ניתוח שגיאות
                    texts = filtered_df.get('text', None)
                    error_analysis = analyzer.advanced_error_analysis(
                        y_true, y_pred, texts, method_name
                    )
                    
                    metrics['error_analysis'] = error_analysis
                    all_metrics[method_name] = metrics
                    
                    print(f"   ✅ Metrics calculated for {len(filtered_df)} samples")
                else:
                    print(f"   ⚠️ No valid labeled data in {method_name}")
            else:
                print(f"   ⚠️ Missing required columns in {method_name}")
        
        if not all_metrics:
            print("❌ No valid metrics calculated!")
            return
        
        # בדיקות מובהקות סטטיסטית
        significance_results = analyzer.statistical_significance_testing(analyzer.all_results)
        
        # יצירת גרפים השוואתיים
        ranking_df = analyzer.create_comprehensive_comparison_plots(all_metrics)
        
        # יצירת דוח סטטיסטי מפורט
        analyzer.generate_statistical_report(all_metrics, significance_results, ranking_df)
        
        print(f"\n✅ Advanced metrics analysis completed successfully!")
        print(f"📊 Results Summary:")
        print(f"   🏆 Best Method: {ranking_df.iloc[0]['Method']} (Score: {ranking_df.iloc[0]['Composite Score']:.4f})")
        print(f"   📈 Methods Analyzed: {len(all_metrics)}")
        print(f"   🔬 Statistical Tests: {len(significance_results)}")
        print(f"   📁 Reports saved to: {analyzer.output_dir}/")
        
        return {
            'metrics': all_metrics,
            'significance': significance_results,
            'ranking': ranking_df,
            'analyzer': analyzer
        }
        
    except Exception as e:
        print(f"❌ Error in advanced metrics analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
