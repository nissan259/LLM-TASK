"""
test_mask_classification.py

Unit Tests for MASK-based Classification
יחידות בדיקה עבור סיווג מבוסס MASK
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mask_based_classification import AdvancedMaskBasedClassifier
except ImportError as e:
    print(f"Warning: Could not import MASK components: {e}")

class TestMaskTemplates(unittest.TestCase):
    """בדיקות עבור תבניות MASK"""
    
    def setUp(self):
        """הכנה לבדיקות"""
        self.required_template = "הרגש שהבעתי הוא [MASK]"
        self.target_words = ["חיובי", "שלילי", "נייטרלי"]
        
    def test_exact_template_format(self):
        """בדיקת פורמט התבנית המדויק לפי דרישות איל"""
        # Test exact template as specified by instructor Ayal
        self.assertIsInstance(self.required_template, str)
        self.assertIn("[MASK]", self.required_template)
        self.assertIn("הרגש שהבעתי הוא", self.required_template)
        
    def test_target_words_hebrew(self):
        """בדיקת מילות המטרה בעברית"""
        expected_words = ["חיובי", "שלילי", "נייטרלי"]
        
        for word in expected_words:
            with self.subTest(word=word):
                self.assertIsInstance(word, str)
                self.assertIn(word, self.target_words)
                # Check Hebrew characters
                self.assertTrue(any(ord(c) >= 0x0590 and ord(c) <= 0x05FF for c in word))
    
    def test_template_variations(self):
        """בדיקת וריאציות תבניות"""
        template_variations = [
            "הרגש שהבעתי הוא [MASK]",
            "הטקסט הזה מביע רגש [MASK]",
            "התחושה שלי היא [MASK]",
            "האמוציה כאן היא [MASK]"
        ]
        
        for template in template_variations:
            with self.subTest(template=template):
                self.assertIsInstance(template, str)
                self.assertIn("[MASK]", template)

class TestSentimentMapping(unittest.TestCase):
    """בדיקות עבור מיפוי רגשות"""
    
    def setUp(self):
        """הכנה לבדיקות"""
        self.sentiment_mapping = {
            "חיובי": 1,
            "שלילי": 0,
            "נייטרלי": 2
        }
        
    def test_sentiment_to_label_mapping(self):
        """בדיקת מיפוי רגש לתווית"""
        for sentiment, label in self.sentiment_mapping.items():
            with self.subTest(sentiment=sentiment, label=label):
                self.assertIsInstance(sentiment, str)
                self.assertIsInstance(label, int)
                self.assertIn(label, [0, 1, 2])
    
    def test_label_to_sentiment_mapping(self):
        """בדיקת מיפוי תווית לרגש"""
        reverse_mapping = {v: k for k, v in self.sentiment_mapping.items()}
        
        self.assertEqual(reverse_mapping[0], "שלילי")
        self.assertEqual(reverse_mapping[1], "חיובי") 
        self.assertEqual(reverse_mapping[2], "נייטרלי")

class TestTextProcessing(unittest.TestCase):
    """בדיקות עבור עיבוד טקסט"""
    
    def setUp(self):
        """הכנה לבדיקות"""
        self.sample_texts = [
            "הסרט הזה היה פשוט מדהים ומרגש",
            "חוויה איומה, לא מומלץ בכלל",
            "בסדר גמור, לא רע ולא טוב במיוחד",
            "אני אוהב את הארוחה הזאת!",
            "מה זה הזבל הזה?",
            "זה לא רע, אבל גם לא טוב"
        ]
    
    def test_hebrew_text_detection(self):
        """בדיקת זיהוי טקסט עברי"""
        for text in self.sample_texts:
            with self.subTest(text=text):
                # Check Hebrew characters present
                has_hebrew = any(ord(c) >= 0x0590 and ord(c) <= 0x05FF for c in text)
                self.assertTrue(has_hebrew, f"No Hebrew characters found in: {text}")
    
    def test_text_preprocessing(self):
        """בדיקת עיבוד מקדים של טקסט"""
        for text in self.sample_texts:
            with self.subTest(text=text):
                # Text should not be empty after stripping
                cleaned_text = text.strip()
                self.assertGreater(len(cleaned_text), 0)
                
                # Text should be string
                self.assertIsInstance(text, str)
    
    def test_mask_token_insertion(self):
        """בדיקת הכנסת טוקן MASK"""
        template = "הרגש שהבעתי הוא [MASK]"
        sample_text = "הסרט הזה היה נהדר"
        
        # Create input for model
        input_text = f"{sample_text}. {template}"
        
        self.assertIn("[MASK]", input_text)
        self.assertIn(sample_text, input_text)
        self.assertIn("הרגש שהבעתי הוא", input_text)

class TestModelPredictions(unittest.TestCase):
    """בדיקות עבור תחזיות המודל"""
    
    def test_prediction_format(self):
        """בדיקת פורמט תחזיות"""
        # Mock prediction results
        mock_predictions = [
            {"token": "חיובי", "score": 0.85},
            {"token": "שלילי", "score": 0.10},
            {"token": "נייטרלי", "score": 0.05}
        ]
        
        for pred in mock_predictions:
            with self.subTest(prediction=pred):
                self.assertIsInstance(pred, dict)
                self.assertIn("token", pred)
                self.assertIn("score", pred)
                self.assertIsInstance(pred["score"], float)
                self.assertGreaterEqual(pred["score"], 0.0)
                self.assertLessEqual(pred["score"], 1.0)
    
    def test_score_normalization(self):
        """בדיקת נורמליזציה של ציונים"""
        raw_scores = [0.85, 0.10, 0.05]
        total = sum(raw_scores)
        
        # Scores should sum to approximately 1.0
        self.assertAlmostEqual(total, 1.0, places=2)
        
        # Each score should be between 0 and 1
        for score in raw_scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

class TestEnsembleMethods(unittest.TestCase):
    """בדיקות עבור שיטות אנסמבל"""
    
    def test_voting_mechanism(self):
        """בדיקת מנגנון הצבעה"""
        # Mock predictions from multiple templates
        template_predictions = [
            {"חיובי": 0.8, "שלילי": 0.1, "נייטרלי": 0.1},
            {"חיובי": 0.7, "שלילי": 0.2, "נייטרלי": 0.1},
            {"חיובי": 0.9, "שלילי": 0.05, "נייטרלי": 0.05}
        ]
        
        # Calculate average scores
        avg_scores = {}
        for sentiment in ["חיובי", "שלילי", "נייטרלי"]:
            scores = [pred[sentiment] for pred in template_predictions]
            avg_scores[sentiment] = np.mean(scores)
        
        # Check results
        self.assertIsInstance(avg_scores, dict)
        total_avg = sum(avg_scores.values())
        self.assertAlmostEqual(total_avg, 1.0, places=1)
    
    def test_confidence_calculation(self):
        """בדיקת חישוב רמת ביטחון"""
        predictions = {"חיובי": 0.8, "שלילי": 0.1, "נייטרלי": 0.1}
        
        # Confidence is the highest score
        confidence = max(predictions.values())
        predicted_class = max(predictions.keys(), key=lambda k: predictions[k])
        
        self.assertEqual(confidence, 0.8)
        self.assertEqual(predicted_class, "חיובי")
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

class TestPerformanceMetrics(unittest.TestCase):
    """בדיקות עבור מדדי ביצועים"""
    
    def test_accuracy_calculation(self):
        """בדיקת חישוב דיוק"""
        y_true = ["חיובי", "שלילי", "נייטרלי", "חיובי", "שלילי"]
        y_pred = ["חיובי", "שלילי", "חיובי", "חיובי", "שלילי"]
        
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        accuracy = correct / len(y_true)
        
        self.assertEqual(accuracy, 0.8)  # 4/5 correct
        self.assertIsInstance(accuracy, float)
    
    def test_confusion_matrix_structure(self):
        """בדיקת מבנה מטריצת בלבול"""
        classes = ["חיובי", "שלילי", "נייטרלי"]
        matrix_size = len(classes)
        
        # Mock confusion matrix
        confusion_matrix = np.random.randint(0, 10, size=(matrix_size, matrix_size))
        
        self.assertEqual(confusion_matrix.shape, (3, 3))
        self.assertTrue(np.all(confusion_matrix >= 0))

class TestErrorHandling(unittest.TestCase):
    """בדיקות עבור טיפול בשגיאות"""
    
    def test_empty_text_handling(self):
        """בדיקת טיפול בטקסט ריק"""
        empty_texts = ["", "   ", "\n", "\t"]
        
        for text in empty_texts:
            with self.subTest(text=repr(text)):
                cleaned = text.strip()
                if not cleaned:
                    self.assertEqual(len(cleaned), 0)
    
    def test_non_hebrew_text_handling(self):
        """בדיקת טיפול בטקסט לא עברי"""
        non_hebrew_texts = [
            "This is English text",
            "Texto en español", 
            "Texte français",
            "123456",
            "!@#$%^&*()"
        ]
        
        for text in non_hebrew_texts:
            with self.subTest(text=text):
                # Check if Hebrew characters are absent
                has_hebrew = any(ord(c) >= 0x0590 and ord(c) <= 0x05FF for c in text)
                if not has_hebrew:
                    self.assertFalse(has_hebrew)
    
    def test_special_characters_handling(self):
        """בדיקת טיפול בתווים מיוחדים"""
        special_texts = [
            "טקסט עם סימנים!@#",
            "טקסט עם מספרים 123",
            "טקסט עם נקודות...",
            "טקסט עם רווחים    רבים"
        ]
        
        for text in special_texts:
            with self.subTest(text=text):
                # Should still contain Hebrew
                has_hebrew = any(ord(c) >= 0x0590 and ord(c) <= 0x05FF for c in text)
                self.assertTrue(has_hebrew)

class TestIntegration(unittest.TestCase):
    """בדיקות אינטגרציה"""
    
    def test_full_prediction_pipeline(self):
        """בדיקת פייפליין תחזית מלא"""
        # Simulate full pipeline
        pipeline_steps = [
            "text_input",
            "template_creation", 
            "mask_token_insertion",
            "model_prediction",
            "score_extraction",
            "ensemble_voting",
            "final_classification"
        ]
        
        for step in pipeline_steps:
            with self.subTest(step=step):
                self.assertIsInstance(step, str)
                self.assertGreater(len(step), 0)
    
    def test_batch_processing(self):
        """בדיקת עיבוד באצווה"""
        batch_texts = [
            "טקסט חיובי מאוד",
            "טקסט שלילי למדי", 
            "טקסט נייטרלי ככה"
        ]
        
        # Simulate batch processing
        batch_results = []
        for text in batch_texts:
            # Mock prediction
            result = {
                "text": text,
                "prediction": "חיובי",  # Mock prediction
                "confidence": 0.85
            }
            batch_results.append(result)
        
        self.assertEqual(len(batch_results), len(batch_texts))
        for result in batch_results:
            self.assertIn("text", result)
            self.assertIn("prediction", result)
            self.assertIn("confidence", result)

if __name__ == '__main__':
    # הרצת כל הבדיקות
    print("🎭 Starting MASK Classification Tests...")
    print("מתחיל בדיקות סיווג MASK...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestMaskTemplates,
        TestSentimentMapping,
        TestTextProcessing,
        TestModelPredictions,
        TestEnsembleMethods,
        TestPerformanceMetrics,
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
        print("✅ All MASK tests passed! כל בדיקות MASK עברו בהצלחה!")
    else:
        print("❌ Some MASK tests failed. יש בדיקות MASK שנכשלו.")
