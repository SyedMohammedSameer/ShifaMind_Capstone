"""
Quick Colab Runner for Comprehensive Evaluation

USAGE IN COLAB:
    1. Replace API_KEY with your OpenAI key
    2. Run this file: !python quick_colab_runner.py

    OR import directly:

    API_KEY = "sk-proj-..."
    from comprehensive_evaluation import run_evaluation
    run_evaluation(api_key=API_KEY, n_samples=30)
"""

import os
import sys

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

API_KEY = "YOUR_API_KEY_HERE"  # <-- Replace with your OpenAI API key
N_SAMPLES = 30  # Number of test samples (more = better stats but slower)

# ============================================================================
# SETUP
# ============================================================================

# Set base path (auto-detects Colab)
os.environ['SHIFAMIND_BASE_PATH'] = '/content/drive/MyDrive/ShifaMind'

# Add repo to path if needed
if '/content/ShifaMind_Capstone' not in sys.path:
    sys.path.append('/content/ShifaMind_Capstone')

# ============================================================================
# RUN
# ============================================================================

if API_KEY == "YOUR_API_KEY_HERE":
    print("=" * 80)
    print("⚠️  ERROR: Please set your OpenAI API key!")
    print("=" * 80)
    print("\nEdit this file and replace:")
    print('  API_KEY = "YOUR_API_KEY_HERE"')
    print("\nWith your actual key:")
    print('  API_KEY = "sk-proj-..."')
    print("\n" + "=" * 80)
    sys.exit(1)

print("🚀 Starting comprehensive evaluation...")
print("=" * 80)

from comprehensive_evaluation import run_evaluation

# Run evaluation
output_path = run_evaluation(api_key=API_KEY, n_samples=N_SAMPLES)

print(f"\n✅ Done! Check {output_path} for outputs")
