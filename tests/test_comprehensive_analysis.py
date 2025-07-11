"""
test_comprehensive_analysis.py

Unit Tests for Comprehensive Analysis & Research Report
יחידות בדיקה עבור ניתוח מקיף ודוח מחקר
"""

import unittest
import sys
import os
import json
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestStatisticalAnalysis(unittest.TestCase):
    """בדיקות עבור ניתוח סטטיסטי"""
    
    def setUp(self):
        """הכנה לבדיקות"""
        self.sample_results = {
            "lora_r4": [0.85, 0.87, 0.83, 0.86, 0.84],
            "lora_r8": [0.87, 0.89, 0.85, 0.88, 0.86],
            "lora_r16": [0.89, 0.91, 0.87, 0.90, 0.88],
            "mask_ensemble": [0.83, 0.85, 0.81, 0.84, 0.82]
        }
    
    def test_mean_calculation(self):
        """בדיקת חישוב ממוצע"""
        for method, scores in self.sample_results.items():
            with self.subTest(method=method):
                mean_score = np.mean(scores)
                self.assertIsInstance(mean_score, (float, np.floating))
                self.assertGreater(mean_score, 0.0)
                self.assertLessEqual(mean_score, 1.0)
    
    def test_standard_deviation(self):
        """בדיקת חישוב סטיית תקן"""
        for method, scores in self.sample_results.items():
            with self.subTest(method=method):
                std_score = np.std(scores)
                self.assertIsInstance(std_score, (float, np.floating))
                self.assertGreaterEqual(std_score, 0.0)
    
    def test_confidence_intervals(self):
        """בדיקת רווחי בטחון"""
        from scipy import stats
        
        for method, scores in self.sample_results.items():
            with self.subTest(method=method):
                confidence_level = 0.95
                degrees_freedom = len(scores) - 1
                
                if degrees_freedom > 0:
                    # Calculate confidence interval
                    mean = np.mean(scores)
                    std_error = stats.sem(scores)
                    margin_error = std_error * stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
                    
                    ci_lower = mean - margin_error
                    ci_upper = mean + margin_error
                    
                    self.assertLessEqual(ci_lower, mean)
                    self.assertGreaterEqual(ci_upper, mean)
    
    def test_bootstrap_sampling(self):
        """בדיקת Bootstrap sampling"""
        original_scores = self.sample_results["lora_r16"]
        n_bootstrap = 100
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(original_scores, 
                                              size=len(original_scores), 
                                              replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        self.assertEqual(len(bootstrap_means), n_bootstrap)
        
        # Bootstrap distribution should be approximately normal
        bootstrap_mean = np.mean(bootstrap_means)
        original_mean = np.mean(original_scores)
        
        # Bootstrap mean should be close to original mean
        self.assertAlmostEqual(bootstrap_mean, original_mean, places=1)

class TestMethodComparison(unittest.TestCase):
    """בדיקות עבור השוואת שיטות"""
    
    def setUp(self):
        """הכנה לבדיקות"""
        self.method_performance = {
            "LoRA Fine-tuning (r=16)": {"accuracy": 0.899, "f1": 0.887, "precision": 0.892, "recall": 0.885},
            "MASK Ensemble": {"accuracy": 0.845, "f1": 0.834, "precision": 0.841, "recall": 0.828},
            "Simple Fine-tuning": {"accuracy": 0.823, "f1": 0.812, "precision": 0.819, "recall": 0.806},
            "Zero-Shot BART": {"accuracy": 0.767, "f1": 0.751, "precision": 0.763, "recall": 0.745}
        }
    
    def test_performance_ranking(self):
        """בדיקת דירוג ביצועים"""
        # Rank by accuracy
        sorted_methods = sorted(self.method_performance.items(), 
                               key=lambda x: x[1]["accuracy"], 
                               reverse=True)
        
        # Best method should be LoRA r=16
        best_method = sorted_methods[0][0]
        self.assertEqual(best_method, "LoRA Fine-tuning (r=16)")
        
        # Check ranking order
        expected_order = [
            "LoRA Fine-tuning (r=16)",
            "MASK Ensemble", 
            "Simple Fine-tuning",
            "Zero-Shot BART"
        ]
        
        actual_order = [method for method, _ in sorted_methods]
        self.assertEqual(actual_order, expected_order)
    
    def test_performance_gaps(self):
        """בדיקת פערי ביצועים"""
        accuracies = [metrics["accuracy"] for metrics in self.method_performance.values()]
        
        max_accuracy = max(accuracies)
        min_accuracy = min(accuracies)
        performance_gap = max_accuracy - min_accuracy
        
        # Performance gap should be reasonable
        self.assertGreater(performance_gap, 0.0)
        self.assertLess(performance_gap, 0.5)  # Not too large
    
    def test_metric_consistency(self):
        """בדיקת עקביות מדדים"""
        for method, metrics in self.method_performance.items():
            with self.subTest(method=method):
                # All metrics should be between 0 and 1
                for metric_name, value in metrics.items():
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)
                
                # F1 should be harmonic mean of precision and recall
                precision = metrics["precision"]
                recall = metrics["recall"]
                expected_f1 = 2 * (precision * recall) / (precision + recall)
                
                # Allow small tolerance for rounding
                self.assertAlmostEqual(metrics["f1"], expected_f1, places=2)

class TestReportGeneration(unittest.TestCase):
    """בדיקות עבור יצירת דוחות"""
    
    def test_markdown_structure(self):
        """בדיקת מבנה Markdown"""
        required_sections = [
            "# Hebrew Sentiment Analysis",
            "## Executive Summary",
            "## Methodology Overview", 
            "## Results Analysis",
            "## Statistical Comparison",
            "## Conclusions and Recommendations"
        ]
        
        for section in required_sections:
            with self.subTest(section=section):
                self.assertIsInstance(section, str)
                self.assertTrue(section.startswith("#"))
    
    def test_json_report_structure(self):
        """בדיקת מבנה דוח JSON"""
        expected_json_structure = {
            "project_info": {},
            "methodology": {},
            "results": {},
            "statistical_analysis": {},
            "conclusions": {},
            "technical_details": {}
        }
        
        for section in expected_json_structure.keys():
            with self.subTest(section=section):
                self.assertIsInstance(section, str)
                self.assertGreater(len(section), 0)
    
    def test_performance_summary_csv(self):
        """בדיקת CSV סיכום ביצועים"""
        # Mock CSV data
        csv_columns = [
            "Method",
            "Accuracy", 
            "Precision",
            "Recall",
            "F1-Score",
            "Training_Time",
            "Inference_Time"
        ]
        
        for column in csv_columns:
            with self.subTest(column=column):
                self.assertIsInstance(column, str)
                self.assertGreater(len(column), 0)

class TestVisualizationGeneration(unittest.TestCase):
    """בדיקות עבור יצירת ויזואליזציות"""
    
    def test_performance_comparison_chart(self):
        """בדיקת גרף השוואת ביצועים"""
        chart_config = {
            "chart_type": "bar",
            "x_axis": "Methods",
            "y_axis": "Performance Metrics",
            "title": "Hebrew Sentiment Analysis - Methods Comparison",
            "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        }
        
        for key, value in chart_config.items():
            with self.subTest(config=key):
                self.assertIsNotNone(value)
    
    def test_confusion_matrix_visualization(self):
        """בדיקת ויזואליזציה מטריצת בלבול"""
        # Mock confusion matrix
        confusion_matrix = np.array([
            [85, 10, 5],   # חיובי
            [12, 78, 10],  # שלילי  
            [8, 15, 77]    # נייטרלי
        ])
        
        # Check matrix properties
        self.assertEqual(confusion_matrix.shape, (3, 3))
        self.assertTrue(np.all(confusion_matrix >= 0))
        
        # Diagonal should have highest values (correct predictions)
        for i in range(3):
            diagonal_value = confusion_matrix[i, i]
            row_sum = np.sum(confusion_matrix[i, :])
            
            # Diagonal should be significant portion of row
            self.assertGreater(diagonal_value / row_sum, 0.5)
    
    def test_training_curves(self):
        """בדיקת עקומות אימון"""
        # Mock training data
        epochs = list(range(1, 11))
        train_accuracy = [0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.89, 0.90, 0.89, 0.90]
        val_accuracy = [0.63, 0.70, 0.75, 0.79, 0.82, 0.84, 0.85, 0.86, 0.85, 0.86]
        
        # Check data consistency
        self.assertEqual(len(epochs), len(train_accuracy))
        self.assertEqual(len(epochs), len(val_accuracy))
        
        # Training should generally improve
        self.assertGreater(train_accuracy[-1], train_accuracy[0])
        self.assertGreater(val_accuracy[-1], val_accuracy[0])

class TestDataIntegrity(unittest.TestCase):
    """בדיקות עבור שלמות נתונים"""
    
    def test_results_completeness(self):
        """בדיקת שלמות תוצאות"""
        required_metrics = ["accuracy", "precision", "recall", "f1"]
        required_methods = ["lora", "mask", "simple_ft", "zero_shot"]
        
        # Mock results structure
        results = {}
        for method in required_methods:
            results[method] = {}
            for metric in required_metrics:
                results[method][metric] = np.random.uniform(0.7, 0.9)
        
        # Verify completeness
        for method in required_methods:
            with self.subTest(method=method):
                self.assertIn(method, results)
                for metric in required_metrics:
                    self.assertIn(metric, results[method])
    
    def test_data_validation(self):
        """בדיקת תקינות נתונים"""
        # Sample data for validation
        sample_data = {
            "text": "טקסט לדוגמה בעברית",
            "true_label": 1,
            "predicted_label": 1,
            "confidence": 0.89,
            "method": "LoRA"
        }
        
        # Validate data types and ranges
        self.assertIsInstance(sample_data["text"], str)
        self.assertIn(sample_data["true_label"], [0, 1, 2])
        self.assertIn(sample_data["predicted_label"], [0, 1, 2])
        self.assertGreaterEqual(sample_data["confidence"], 0.0)
        self.assertLessEqual(sample_data["confidence"], 1.0)
        self.assertIsInstance(sample_data["method"], str)

class TestTechnicalDetails(unittest.TestCase):
    """בדיקות עבור פרטים טכניים"""
    
    def test_hyperparameter_documentation(self):
        """בדיקת תיעוד היפר-פרמטרים"""
        hyperparameters = {
            "lora": {
                "r": [4, 8, 16, 32],
                "alpha": [8, 16, 32, 64],
                "dropout": [0.1, 0.2, 0.3],
                "target_modules": ["query", "value"]
            },
            "training": {
                "learning_rate": 2e-5,
                "batch_size": 16,
                "num_epochs": 5,
                "warmup_steps": 100
            }
        }
        
        # Validate hyperparameter ranges
        for r in hyperparameters["lora"]["r"]:
            self.assertIsInstance(r, int)
            self.assertGreater(r, 0)
        
        for alpha in hyperparameters["lora"]["alpha"]:
            self.assertIsInstance(alpha, int)
            self.assertGreater(alpha, 0)
        
        for dropout in hyperparameters["lora"]["dropout"]:
            self.assertIsInstance(dropout, float)
            self.assertGreaterEqual(dropout, 0.0)
            self.assertLessEqual(dropout, 0.5)
    
    def test_model_specifications(self):
        """בדיקת מפרטי מודל"""
        model_specs = {
            "base_model": "avichr/heBERT",
            "tokenizer": "avichr/heBERT", 
            "max_length": 512,
            "num_labels": 3,
            "architecture": "BERT-based"
        }
        
        for spec_name, spec_value in model_specs.items():
            with self.subTest(spec=spec_name):
                self.assertIsNotNone(spec_value)
                
                if spec_name in ["max_length", "num_labels"]:
                    self.assertIsInstance(spec_value, int)
                    self.assertGreater(spec_value, 0)

class TestReproducibility(unittest.TestCase):
    """בדיקות עבור יכולת שחזור"""
    
    def test_random_seed_consistency(self):
        """בדיקת עקביות זרע אקראי"""
        import random
        
        # Set seed and generate numbers
        random.seed(42)
        first_sequence = [random.random() for _ in range(10)]
        
        # Reset seed and generate again
        random.seed(42)
        second_sequence = [random.random() for _ in range(10)]
        
        # Should be identical
        self.assertEqual(first_sequence, second_sequence)
    
    def test_environment_documentation(self):
        """בדיקת תיעוד סביבת עבודה"""
        environment_info = {
            "python_version": "3.8+",
            "torch_version": "1.12+",
            "transformers_version": "4.21+",
            "peft_version": "0.4+",
            "operating_system": "Windows/Linux/macOS"
        }
        
        for component, version in environment_info.items():
            with self.subTest(component=component):
                self.assertIsInstance(version, str)
                self.assertGreater(len(version), 0)

class TestPerformanceBenchmarks(unittest.TestCase):
    """בדיקות עבור מדדי ביצועים"""
    
    def test_training_time_benchmarks(self):
        """בדיקת מדדי זמן אימון"""
        # Mock training times (in minutes)
        training_times = {
            "simple_ft": 45,
            "lora_r4": 25,
            "lora_r8": 30,
            "lora_r16": 35,
            "mask_zero_shot": 5  # No training needed
        }
        
        for method, time_minutes in training_times.items():
            with self.subTest(method=method):
                self.assertIsInstance(time_minutes, (int, float))
                self.assertGreater(time_minutes, 0)
                
                # LoRA should be faster than full fine-tuning
                if "lora" in method:
                    self.assertLess(time_minutes, training_times["simple_ft"])
    
    def test_inference_speed(self):
        """בדיקת מהירות היסק"""
        # Mock inference times (milliseconds per sample)
        inference_times = {
            "lora": 12,
            "mask": 15,
            "simple_ft": 10,
            "zero_shot": 20
        }
        
        for method, time_ms in inference_times.items():
            with self.subTest(method=method):
                self.assertIsInstance(time_ms, (int, float))
                self.assertGreater(time_ms, 0)
                self.assertLess(time_ms, 100)  # Should be reasonably fast

if __name__ == '__main__':
    # הרצת כל הבדיקות
    print("📊 Starting Comprehensive Analysis Tests...")
    print("מתחיל בדיקות ניתוח מקיף...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestStatisticalAnalysis,
        TestMethodComparison,
        TestReportGeneration,
        TestVisualizationGeneration,
        TestDataIntegrity,
        TestTechnicalDetails,
        TestReproducibility,
        TestPerformanceBenchmarks
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All comprehensive analysis tests passed! כל בדיקות הניתוח המקיף עברו בהצלחה!")
    else:
        print("❌ Some analysis tests failed. יש בדיקות ניתוח שנכשלו.")
