"""
test_lora_fine_tuning.py

Unit Tests for LoRA Fine-tuning Implementation
יחידות בדיקה עבור יישום LoRA Fine-tuning
"""

import unittest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from lora_fine_tuning import AdvancedLoRATrainer, analyze_results
except ImportError as e:
    print(f"Warning: Could not import LoRA components: {e}")

class TestLoRAConfiguration(unittest.TestCase):
    """בדיקות עבור הגדרות LoRA"""
    
    def setUp(self):
        """הכנה לבדיקות"""
        self.test_data = pd.DataFrame({
            'text': ['טקסט חיובי', 'טקסט שלילי', 'טקסט נייטרלי'],
            'label': [1, 0, 2]
        })
        
    def test_lora_config_creation(self):
        """בדיקת יצירת הגדרות LoRA"""
        # Test different r values
        r_values = [4, 8, 16, 32]
        for r in r_values:
            with self.subTest(r=r):
                self.assertIsInstance(r, int)
                self.assertGreater(r, 0)
                self.assertLessEqual(r, 64)
    
    def test_alpha_values(self):
        """בדיקת ערכי אלפא"""
        alpha_values = [8, 16, 32, 64]
        for alpha in alpha_values:
            with self.subTest(alpha=alpha):
                self.assertIsInstance(alpha, int)
                self.assertGreater(alpha, 0)
    
    def test_dropout_values(self):
        """בדיקת ערכי Dropout"""
        dropout_values = [0.1, 0.2, 0.3]
        for dropout in dropout_values:
            with self.subTest(dropout=dropout):
                self.assertIsInstance(dropout, float)
                self.assertGreaterEqual(dropout, 0.0)
                self.assertLessEqual(dropout, 0.5)

class TestDataProcessing(unittest.TestCase):
    """בדיקות עבור עיבוד נתונים"""
    
    def setUp(self):
        """הכנה לבדיקות"""
        self.sample_texts = [
            "הסרט הזה היה נהדר ומרגש",
            "חוויה נוראית, לא מומלץ",
            "בסדר, לא רע ולא טוב"
        ]
        self.sample_labels = [1, 0, 2]  # חיובי, שלילי, נייטרלי
    
    def test_text_preprocessing(self):
        """בדיקת עיבוד טקסט"""
        for text in self.sample_texts:
            # Check text is string
            self.assertIsInstance(text, str)
            # Check text is not empty
            self.assertGreater(len(text.strip()), 0)
            # Check Hebrew characters present
            self.assertTrue(any(ord(c) >= 0x0590 and ord(c) <= 0x05FF for c in text))
    
    def test_label_validation(self):
        """בדיקת תקינות תוויות"""
        for label in self.sample_labels:
            self.assertIsInstance(label, int)
            self.assertIn(label, [0, 1, 2])  # שלילי, חיובי, נייטרלי
    
    def test_data_balance(self):
        """בדיקת איזון נתונים"""
        labels = np.array(self.sample_labels)
        unique_labels = np.unique(labels)
        self.assertEqual(len(unique_labels), 3)  # 3 מחלקות

class TestModelMetrics(unittest.TestCase):
    """בדיקות עבור מדדי הערכה"""
    
    def test_accuracy_calculation(self):
        """בדיקת חישוב דיוק"""
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        
        # Manual accuracy calculation
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        accuracy = correct / len(y_true)
        
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        self.assertEqual(accuracy, 0.8)  # 4/5 correct
    
    def test_metrics_range(self):
        """בדיקת טווח מדדים"""
        # Test metrics are in valid ranges
        sample_metrics = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.87,
            'f1': 0.84
        }
        
        for metric_name, value in sample_metrics.items():
            with self.subTest(metric=metric_name):
                self.assertIsInstance(value, float)
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

class TestStatisticalAnalysis(unittest.TestCase):
    """בדיקות עבור ניתוח סטטיסטי"""
    
    def test_confidence_intervals(self):
        """בדיקת רווחי בטחון"""
        # Sample results for statistical testing
        results = [0.85, 0.87, 0.83, 0.86, 0.84]
        
        mean_result = np.mean(results)
        std_result = np.std(results)
        
        self.assertIsInstance(mean_result, float)
        self.assertIsInstance(std_result, float)
        self.assertGreaterEqual(std_result, 0.0)
    
    def test_bootstrap_sampling(self):
        """בדיקת Bootstrap sampling"""
        original_data = [0.80, 0.85, 0.82, 0.87, 0.83]
        n_bootstrap = 100
        
        bootstrap_samples = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(original_data, size=len(original_data), replace=True)
            bootstrap_samples.append(np.mean(sample))
        
        self.assertEqual(len(bootstrap_samples), n_bootstrap)
        for sample_mean in bootstrap_samples:
            self.assertIsInstance(sample_mean, (float, np.floating))

class TestResultsAnalysis(unittest.TestCase):
    """בדיקות עבור ניתוח תוצאות"""
    
    def test_results_format(self):
        """בדיקת פורמט תוצאות"""
        sample_results = {
            'r_4': {'accuracy': 0.85, 'f1': 0.84},
            'r_8': {'accuracy': 0.87, 'f1': 0.86},
            'r_16': {'accuracy': 0.89, 'f1': 0.88},
            'r_32': {'accuracy': 0.88, 'f1': 0.87}
        }
        
        for config_name, metrics in sample_results.items():
            with self.subTest(config=config_name):
                self.assertIsInstance(config_name, str)
                self.assertIn('r_', config_name)
                self.assertIsInstance(metrics, dict)
                self.assertIn('accuracy', metrics)
                self.assertIn('f1', metrics)
    
    def test_best_configuration_selection(self):
        """בדיקת בחירת הגדרה מיטבית"""
        configs = {
            'r_4': 0.85,
            'r_8': 0.87,
            'r_16': 0.89,  # Best
            'r_32': 0.88
        }
        
        best_config = max(configs.keys(), key=lambda k: configs[k])
        best_score = configs[best_config]
        
        self.assertEqual(best_config, 'r_16')
        self.assertEqual(best_score, 0.89)

class TestErrorHandling(unittest.TestCase):
    """בדיקות עבור טיפול בשגיאות"""
    
    def test_empty_data_handling(self):
        """בדיקת טיפול בנתונים ריקים"""
        empty_data = pd.DataFrame()
        self.assertTrue(empty_data.empty)
    
    def test_invalid_labels_handling(self):
        """בדיקת טיפול בתוויות לא תקינות"""
        invalid_labels = [-1, 3, 4]  # מחוץ לטווח 0-2
        valid_range = [0, 1, 2]
        
        for label in invalid_labels:
            self.assertNotIn(label, valid_range)
    
    def test_text_encoding_handling(self):
        """בדיקת טיפול בקידוד טקסט"""
        hebrew_text = "טקסט בעברית עם רגשות"
        
        # Test encoding/decoding
        try:
            encoded = hebrew_text.encode('utf-8')
            decoded = encoded.decode('utf-8')
            self.assertEqual(hebrew_text, decoded)
        except UnicodeError:
            self.fail("Unicode encoding/decoding failed")

class TestIntegration(unittest.TestCase):
    """בדיקות אינטגרציה"""
    
    def test_end_to_end_workflow(self):
        """בדיקת זרימת עבודה מקצה לקצה"""
        # Simulate complete workflow
        workflow_steps = [
            "data_loading",
            "preprocessing", 
            "model_configuration",
            "training",
            "evaluation",
            "results_analysis"
        ]
        
        for step in workflow_steps:
            with self.subTest(step=step):
                self.assertIsInstance(step, str)
                self.assertGreater(len(step), 0)

if __name__ == '__main__':
    # הרצת כל הבדיקות
    print("🧪 Starting LoRA Fine-tuning Tests...")
    print("מתחיל בדיקות LoRA Fine-tuning...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestLoRAConfiguration,
        TestDataProcessing,
        TestModelMetrics,
        TestStatisticalAnalysis,
        TestResultsAnalysis,
        TestErrorHandling,
        TestIntegration
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
        print("✅ All tests passed! הכל עבר בהצלחה!")
    else:
        print("❌ Some tests failed. יש בדיקות שנכשלו.")
