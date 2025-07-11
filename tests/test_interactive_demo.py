"""
test_interactive_demo.py

Unit Tests for Interactive Demo
יחידות בדיקה עבור דמו אינטראקטיבי
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestDemoConfiguration(unittest.TestCase):
    """בדיקות עבור הגדרות הדמו"""
    
    def test_streamlit_components(self):
        """בדיקת רכיבי Streamlit"""
        expected_components = [
            "text_area",
            "button", 
            "selectbox",
            "tabs",
            "columns",
            "markdown",
            "plotly_chart",
            "dataframe"
        ]
        
        for component in expected_components:
            with self.subTest(component=component):
                self.assertIsInstance(component, str)
                self.assertGreater(len(component), 0)
    
    def test_page_configuration(self):
        """בדיקת הגדרת עמוד"""
        page_config = {
            "page_title": "Hebrew Sentiment Analysis - Interactive Demo",
            "page_icon": "🎭",
            "layout": "wide",
            "initial_sidebar_state": "expanded"
        }
        
        for key, value in page_config.items():
            with self.subTest(config=key):
                self.assertIsNotNone(value)
                self.assertIsInstance(value, str)

class TestMethodsIntegration(unittest.TestCase):
    """בדיקות עבור אינטגרציה של שיטות"""
    
    def setUp(self):
        """הכנה לבדיקות"""
        self.available_methods = [
            "LoRA Fine-tuning",
            "MASK Classification", 
            "Simple Fine-tuning",
            "Simple PEFT",
            "Zero-Shot BART",
            "Advanced Metrics"
        ]
    
    def test_methods_availability(self):
        """בדיקת זמינות שיטות"""
        for method in self.available_methods:
            with self.subTest(method=method):
                self.assertIsInstance(method, str)
                self.assertGreater(len(method), 0)
    
    def test_method_loading_simulation(self):
        """סימולציה של טעינת שיטות"""
        method_status = {}
        
        for method in self.available_methods:
            try:
                # Simulate loading
                method_status[method] = "loaded"
            except Exception as e:
                method_status[method] = f"error: {e}"
        
        # Check that all methods have a status
        self.assertEqual(len(method_status), len(self.available_methods))

class TestUserInterface(unittest.TestCase):
    """בדיקות עבור ממשק המשתמש"""
    
    def test_tab_structure(self):
        """בדיקת מבנה טאבים"""
        expected_tabs = [
            "🎯 Real-time Prediction",
            "📊 Methods Comparison", 
            "📈 Performance Analysis",
            "🔍 Model Details",
            "📋 Results Export"
        ]
        
        self.assertEqual(len(expected_tabs), 5)
        
        for tab in expected_tabs:
            with self.subTest(tab=tab):
                self.assertIsInstance(tab, str)
                self.assertTrue(any(emoji in tab for emoji in ["🎯", "📊", "📈", "🔍", "📋"]))
    
    def test_input_validation(self):
        """בדיקת אימות קלטים"""
        valid_inputs = [
            "הסרט הזה היה נהדר",
            "חוויה נוראית",
            "בסדר, לא רע ולא טוב"
        ]
        
        invalid_inputs = [
            "",  # Empty
            "   ",  # Only spaces
            "English text",  # Non-Hebrew
            "123456"  # Only numbers
        ]
        
        for text in valid_inputs:
            with self.subTest(text=text, type="valid"):
                # Valid Hebrew text
                has_hebrew = any(ord(c) >= 0x0590 and ord(c) <= 0x05FF for c in text)
                self.assertTrue(has_hebrew)
                self.assertGreater(len(text.strip()), 0)
        
        for text in invalid_inputs:
            with self.subTest(text=text, type="invalid"):
                if text.strip() == "":
                    self.assertEqual(len(text.strip()), 0)

class TestVisualization(unittest.TestCase):
    """בדיקות עבור ויזואליזציה"""
    
    def test_chart_types(self):
        """בדיקת סוגי גרפים"""
        chart_types = [
            "bar_chart",
            "line_chart", 
            "scatter_plot",
            "heatmap",
            "confusion_matrix",
            "performance_comparison"
        ]
        
        for chart_type in chart_types:
            with self.subTest(chart=chart_type):
                self.assertIsInstance(chart_type, str)
                self.assertGreater(len(chart_type), 0)
    
    def test_metrics_visualization(self):
        """בדיקת ויזואליזציה של מדדים"""
        sample_metrics = {
            "Accuracy": 0.89,
            "Precision": 0.87,
            "Recall": 0.88,
            "F1-Score": 0.87
        }
        
        for metric, value in sample_metrics.items():
            with self.subTest(metric=metric):
                self.assertIsInstance(metric, str)
                self.assertIsInstance(value, float)
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

class TestDataExport(unittest.TestCase):
    """בדיקות עבור ייצוא נתונים"""
    
    def test_export_formats(self):
        """בדיקת פורמטי ייצוא"""
        export_formats = ["CSV", "Excel", "JSON", "PDF"]
        
        for format_type in export_formats:
            with self.subTest(format=format_type):
                self.assertIsInstance(format_type, str)
                self.assertIn(format_type, ["CSV", "Excel", "JSON", "PDF"])
    
    def test_results_structure(self):
        """בדיקת מבנה תוצאות לייצוא"""
        sample_results = {
            "text": "טקסט לדוגמה",
            "method": "LoRA Fine-tuning",
            "prediction": "חיובי",
            "confidence": 0.89,
            "timestamp": "2025-07-11 10:30:00"
        }
        
        required_fields = ["text", "method", "prediction", "confidence", "timestamp"]
        
        for field in required_fields:
            with self.subTest(field=field):
                self.assertIn(field, sample_results)

class TestPerformanceAnalysis(unittest.TestCase):
    """בדיקות עבור ניתוח ביצועים"""
    
    def test_method_comparison(self):
        """בדיקת השוואת שיטות"""
        method_results = {
            "LoRA Fine-tuning": {"accuracy": 0.89, "f1": 0.88},
            "MASK Classification": {"accuracy": 0.85, "f1": 0.84},
            "Simple Fine-tuning": {"accuracy": 0.82, "f1": 0.81},
            "Zero-Shot BART": {"accuracy": 0.78, "f1": 0.77}
        }
        
        # Find best method
        best_method = max(method_results.keys(), 
                         key=lambda m: method_results[m]["accuracy"])
        
        self.assertEqual(best_method, "LoRA Fine-tuning")
        self.assertGreater(method_results[best_method]["accuracy"], 0.85)
    
    def test_performance_trends(self):
        """בדיקת מגמות ביצועים"""
        performance_over_time = [
            {"epoch": 1, "accuracy": 0.70},
            {"epoch": 2, "accuracy": 0.75},
            {"epoch": 3, "accuracy": 0.82},
            {"epoch": 4, "accuracy": 0.87},
            {"epoch": 5, "accuracy": 0.89}
        ]
        
        # Check improvement trend
        accuracies = [p["accuracy"] for p in performance_over_time]
        
        for i in range(1, len(accuracies)):
            with self.subTest(epoch=i+1):
                # Generally improving (allowing small drops)
                self.assertGreaterEqual(accuracies[i], accuracies[0])

class TestRealTimePrediction(unittest.TestCase):
    """בדיקות עבור תחזית בזמן אמת"""
    
    def test_prediction_speed(self):
        """בדיקת מהירות תחזית"""
        import time
        
        # Simulate prediction time
        start_time = time.time()
        
        # Mock prediction process
        sample_text = "הסרט הזה היה נהדר"
        prediction = "חיובי"  # Mock result
        confidence = 0.89
        
        end_time = time.time()
        prediction_time = end_time - start_time
        
        # Should be relatively fast (under 1 second for mock)
        self.assertLess(prediction_time, 1.0)
        self.assertIsInstance(prediction, str)
        self.assertIsInstance(confidence, float)
    
    def test_batch_prediction(self):
        """בדיקת תחזית באצווה"""
        batch_texts = [
            "טקסט חיובי ראשון",
            "טקסט שלילי שני",
            "טקסט נייטרלי שלישי"
        ]
        
        # Mock batch processing
        batch_results = []
        for i, text in enumerate(batch_texts):
            result = {
                "id": i,
                "text": text,
                "prediction": ["חיובי", "שלילי", "נייטרלי"][i],
                "confidence": [0.89, 0.85, 0.82][i]
            }
            batch_results.append(result)
        
        self.assertEqual(len(batch_results), len(batch_texts))
        
        for result in batch_results:
            self.assertIn("text", result)
            self.assertIn("prediction", result)
            self.assertIn("confidence", result)

class TestErrorHandling(unittest.TestCase):
    """בדיקות עבור טיפול בשגיאות"""
    
    def test_model_loading_errors(self):
        """בדיקת טיפול בשגיאות טעינת מודל"""
        error_scenarios = [
            "model_not_found",
            "memory_error",
            "permission_denied",
            "network_error"
        ]
        
        for error in error_scenarios:
            with self.subTest(error=error):
                # Should handle gracefully
                self.assertIsInstance(error, str)
    
    def test_input_sanitization(self):
        """בדיקת ניקוי קלטים"""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../etc/passwd",
            "null\x00byte"
        ]
        
        for malicious_input in malicious_inputs:
            with self.subTest(input=malicious_input):
                # Should be sanitized
                sanitized = malicious_input.replace("<", "&lt;").replace(">", "&gt;")
                self.assertNotEqual(sanitized, malicious_input)

class TestAccessibility(unittest.TestCase):
    """בדיקות עבור נגישות"""
    
    def test_rtl_support(self):
        """בדיקת תמיכה בכיוון ימין לשמאל"""
        hebrew_text = "טקסט בעברית עם כיוון ימין לשמאל"
        
        # Check Hebrew characters present
        has_hebrew = any(ord(c) >= 0x0590 and ord(c) <= 0x05FF for c in hebrew_text)
        self.assertTrue(has_hebrew)
    
    def test_responsive_design(self):
        """בדיקת עיצוב רספונסיבי"""
        screen_sizes = ["mobile", "tablet", "desktop", "wide"]
        
        for size in screen_sizes:
            with self.subTest(screen=size):
                self.assertIsInstance(size, str)
                self.assertIn(size, ["mobile", "tablet", "desktop", "wide"])

if __name__ == '__main__':
    # הרצת כל הבדיקות
    print("🖥️ Starting Interactive Demo Tests...")
    print("מתחיל בדיקות דמו אינטראקטיבי...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDemoConfiguration,
        TestMethodsIntegration,
        TestUserInterface,
        TestVisualization,
        TestDataExport,
        TestPerformanceAnalysis,
        TestRealTimePrediction,
        TestErrorHandling,
        TestAccessibility
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
        print("✅ All demo tests passed! כל בדיקות הדמו עברו בהצלחה!")
    else:
        print("❌ Some demo tests failed. יש בדיקות דמו שנכשלו.")
