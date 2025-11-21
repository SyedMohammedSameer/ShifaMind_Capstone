# ShifaMind Comprehensive Evaluation Guide

## 🚀 Quick Start (Google Colab)

### Method 1: Direct Import (Recommended)

```python
# 1. Set your API key
API_KEY = "sk-proj-YOUR_KEY_HERE"

# 2. Import and run
from comprehensive_evaluation import run_evaluation

# 3. Run evaluation (30 samples recommended)
run_evaluation(api_key=API_KEY, n_samples=30)
```

### Method 2: Using Quick Runner Script

```python
# 1. Edit quick_colab_runner.py and add your API key
# 2. Run it
!python quick_colab_runner.py
```

---

## 📊 What Gets Generated

### Visualizations (PNG, 300 DPI)
1. **inference_time_comparison.png** - Speed comparison bar chart
2. **cost_comparison.png** - Cost per 1k predictions
3. **confidence_distributions.png** - Prediction confidence histograms
4. **capabilities_matrix.png** - Feature comparison heatmap

### Tables (CSV)
- **model_comparison_table.csv** - Summary metrics for report

### Diagrams (Markdown)
- **architecture_diagram.md** - ShifaMind architecture (Mermaid)

### Data (JSON)
- **comparison_results.json** - Raw prediction data

---

## ⚙️ Configuration

```python
# Adjust number of samples (more = better stats, slower runtime)
run_evaluation(api_key=API_KEY, n_samples=50)

# Set custom output path
import os
os.environ['SHIFAMIND_BASE_PATH'] = '/your/custom/path'
```

---

## 📈 Expected Runtime

| Component | Time (30 samples) |
|-----------|-------------------|
| ShifaMind predictions | ~3 seconds |
| Bio_ClinicalBERT | ~3 seconds |
| GPT-4o-mini API calls | ~1-2 minutes |
| Visualization generation | ~5 seconds |
| **Total** | **~2-3 minutes** |

---

## 💰 Cost Estimate

- **ShifaMind**: $0 (runs locally)
- **Bio_ClinicalBERT**: $0 (runs locally)
- **GPT-4o-mini**: ~$0.02-0.05 per 30 samples

---

## 🔧 Troubleshooting

### "Module not found"
```python
import sys
sys.path.append('/content/ShifaMind_Capstone')
```

### "API key invalid"
- Check your key starts with `sk-proj-`
- Verify it's not expired

### "Out of memory"
- Reduce `n_samples` to 10-20
- Restart Colab runtime

### "Path not found"
```python
import os
os.environ['SHIFAMIND_BASE_PATH'] = '/content/drive/MyDrive/ShifaMind'
```

---

## 📊 Output Location

All results saved to:
```
/content/drive/MyDrive/ShifaMind/05_Comparisons/
```

---

## 🎯 For Your Capstone

### Use These in Your Poster:
- `capabilities_matrix.png` - Shows ShifaMind superiority
- `cost_comparison.png` - FREE vs expensive
- `inference_time_comparison.png` - Speed metrics

### Use These in Your Report:
- `model_comparison_table.csv` - Quantitative results
- `architecture_diagram.md` - Technical architecture

---

## 📝 Citation

If you modify this evaluation framework, update this README accordingly.

---

**Author**: Mohammed Sameer Syed
**Date**: November 2025
**Institution**: University of Arizona
