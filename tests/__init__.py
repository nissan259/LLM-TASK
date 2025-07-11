"""
__init__.py

Tests Package Initialization
אתחול חבילת הבדיקות
"""

# Test package for Hebrew Sentiment Analysis Project
# חבילת בדיקות עבור פרויקט ניתוח רגשות עברית

__version__ = "1.0.0"
__author__ = "Ben & Oral"
__description__ = "Comprehensive test suite for Hebrew sentiment analysis methods"

# Test categories
TEST_CATEGORIES = {
    "lora": "LoRA Fine-tuning Tests",
    "mask": "MASK Classification Tests", 
    "demo": "Interactive Demo Tests",
    "analysis": "Comprehensive Analysis Tests"
}

# Test configuration
TEST_CONFIG = {
    "verbose": True,
    "generate_reports": True,
    "save_json": True,
    "timeout_seconds": 300
}

print("🧪 Hebrew Sentiment Analysis Test Suite Initialized")
print("חבילת בדיקות ניתוח רגשות עברית אותחלה")
