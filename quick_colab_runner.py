"""
Quick Colab Runner for Comprehensive Evaluation

Simplified script to run in Google Colab with minimal setup.
"""

import os

# Set your OpenAI API key here
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your key

# Set base path
os.environ['SHIFAMIND_BASE_PATH'] = '/content/drive/MyDrive/ShifaMind'

# Run comprehensive evaluation
print("🚀 Starting comprehensive evaluation...")
print("=" * 80)

# Import and run
import sys
sys.path.append('/content/ShifaMind_Capstone')

from comprehensive_evaluation import main

# Run with 30 samples (adjust as needed)
# More samples = better statistics but longer runtime
main(api_key=API_KEY, n_samples=30)

print("\n✅ Done! Check /content/drive/MyDrive/ShifaMind/05_Comparisons for outputs")
