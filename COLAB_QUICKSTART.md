# 🚀 ShifaMind - Google Colab Quick Start Guide

**Run ShifaMind on Google Colab with ZERO local setup!**

---

## ⚡ Ultra-Quick Start (3 Steps)

1. **Upload your data to Google Drive** at: `My Drive/ShifaMind/01_Raw_Datasets/`
2. **Open** `colab_setup.ipynb` in Google Colab
3. **Run all cells** - Done! ✅

---

## 📋 Prerequisites

### Data Requirements

Your Google Drive should have this structure:

```
My Drive/
└── ShifaMind/
    └── 01_Raw_Datasets/
        └── Extracted/
            ├── umls-2025AA-metathesaurus-full/
            │   └── 2025AA/META/
            │       ├── MRCONSO.RRF
            │       ├── MRDEF.RRF
            │       └── MRSTY.RRF
            ├── icd10cm-CodesDescriptions-2024/
            │   └── icd10cm-codes-2024.txt
            ├── mimic-iv-3.1/
            │   └── (MIMIC-IV files)
            └── mimic-iv-note-2.2/note/
                └── discharge.csv.gz
```

### Where to Get the Data

1. **UMLS**: Register at https://uts.nlm.nih.gov/ (free, requires account)
2. **MIMIC-IV**: Complete training at https://physionet.org/ (free, requires credentials)
3. **ICD-10**: Download from https://www.cms.gov/ (public, no registration)

---

## 🎯 Step-by-Step Instructions

### Step 1: Prepare Google Drive

1. Upload all data files to your Google Drive
2. Ensure folder names match exactly (case-sensitive!)
3. Verify files are extracted (not .zip or .tar.gz)

### Step 2: Enable GPU in Colab

1. Open Google Colab
2. Go to **Runtime → Change runtime type**
3. Select **T4 GPU** (free) or **A100 GPU** (with Colab Pro)
4. Click **Save**

### Step 3: Open the Setup Notebook

**Option A: Direct Upload**
- Download `colab_setup.ipynb` from the repo
- Upload to Google Colab

**Option B: From GitHub** (if public)
- Click: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SyedMohammedSameer/ShifaMind_Capstone/blob/main/colab_setup.ipynb)

### Step 4: Run the Notebook

1. Click **Runtime → Run all** (or press Ctrl+F9)
2. When prompted, authorize Google Drive access
3. Wait for cells to execute (progress bars will show)

The notebook will automatically:
- ✅ Mount Google Drive
- ✅ Clone the repository
- ✅ Install dependencies
- ✅ Validate data paths
- ✅ Configure environment

---

## 🛠️ Custom Paths (Optional)

### If Your Data is Elsewhere

**In the notebook**, modify Step 3 to set your custom path:

```python
# Change this line:
BASE_PATH = "/content/drive/MyDrive/ShifaMind"

# To your custom location:
BASE_PATH = "/content/drive/MyDrive/YourFolder/ShifaMind"
```

### Using the Path Helper

```python
# Validate paths
!python setup_paths.py --base-path /content/drive/MyDrive/YourFolder/ShifaMind

# Just validate, don't set
!python setup_paths.py --base-path /path/to/data --no-set
```

---

## 📊 Pipeline Overview

The notebook runs these steps in order:

| Step | Script | Time | Description |
|------|--------|------|-------------|
| 1 | `final_knowledge_base_generator.py` | ~10 min | Parse UMLS/ICD-10 → JSON |
| 2 | `final_model_training.py` | ~6 hrs | Train model (3 stages) |
| 3 | `final_evaluation.py` | ~15 min | Compute metrics + plots |
| 4 | `final_demo.py` | instant | Launch web interface |

**Total Time**: ~6-7 hours on T4 GPU, ~1-2 hours on A100 GPU

---

## ⏱️ Time Estimates

### By GPU Type

| GPU Type | Availability | Training Time | Cost |
|----------|-------------|---------------|------|
| **T4** | Free (Colab) | ~6 hours | Free |
| **A100** | Colab Pro | ~1 hour | $9.99/month |
| **CPU** | Free (Colab) | ~48 hours | Free (not recommended) |

### By Stage

| Stage | Operation | Time (T4) |
|-------|-----------|-----------|
| Setup | Install deps + mount drive | ~3 min |
| KB Generation | Parse UMLS/ICD-10 | ~10 min |
| Stage 1 Training | Diagnosis head | ~2 hrs |
| Stage 2 Training | Concept head | ~1.5 hrs |
| Stage 3 Training | Joint fine-tuning | ~2.5 hrs |
| Evaluation | Metrics + plots | ~15 min |

---

## 💡 Tips & Tricks

### Avoid Session Timeout

Colab free tier disconnects after:
- **30 minutes** of inactivity
- **12 hours** maximum runtime

**Solutions**:
1. Keep browser tab active
2. Periodically click in the notebook
3. Use Colab Pro (24-hour runtime)
4. Run training overnight with tab open

### Save Your Work

The notebook automatically backs up models to Google Drive:
- ✅ Checkpoints saved after each stage
- ✅ Final model saved to Drive
- ✅ Safe even if Colab disconnects

### Check GPU Memory

```python
import torch
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

### Speed Up Training (Demo Mode)

For testing, use a smaller dataset:

```python
# In the training cell, add this flag:
!python final_model_training.py --demo
```

This trains on 1000 samples instead of 3000 (~2x faster).

---

## 🐛 Troubleshooting

### "Path not found" Error

**Problem**: Can't find UMLS or MIMIC files

**Solution**:
1. Verify files are in Google Drive
2. Check folder names match exactly (case-sensitive!)
3. Run `!python setup_paths.py` to validate
4. Re-run Step 3 in the notebook

### "Out of Memory" Error

**Problem**: GPU runs out of VRAM during training

**Solutions**:
- Reduce batch size: Edit `config.py` → set `BATCH_SIZE = 4`
- Use CPU (slower): Runtime → Change runtime type → None
- Upgrade to A100 GPU (Colab Pro)

### "Session Disconnected" Error

**Problem**: Colab timed out during training

**Solution**:
- Model checkpoints are saved after each stage
- Just re-run the training cell - it will resume from last checkpoint
- Or use Colab Pro for longer sessions

### "Module not found" Error

**Problem**: Missing Python package

**Solution**:
```python
# Install missing packages
!pip install transformers gradio scispacy
!python -m spacy download en_core_sci_sm
```

### GPU Not Available

**Problem**: `torch.cuda.is_available()` returns `False`

**Solution**:
1. Go to **Runtime → Change runtime type**
2. Select **T4 GPU** (or A100 with Pro)
3. Click **Save**
4. **Restart runtime** (Runtime → Restart runtime)
5. Re-run from Step 1

---

## 🎓 After Training

### Use the Model

```python
from final_inference import ShifaMindPredictor

# Load trained model
predictor = ShifaMindPredictor()

# Predict on clinical note
result = predictor.predict("""
Patient presents with fever, cough, and dyspnea.
Chest X-ray shows right lower lobe infiltrate.
""")

print(result['diagnosis']['name'])  # Pneumonia, unspecified organism
```

### Launch Demo Interface

```python
# Run in a notebook cell
!python final_demo.py
```

This creates a Gradio web interface with:
- Real-time prediction
- Evidence highlighting
- JSON export
- Public sharing link (72-hour validity)

### Export Results

All results are saved to Google Drive:
- `03_Models/checkpoints/` - Model weights
- `04_Results/experiments/` - Metrics, plots, examples

### Download Model

```python
# Download to local machine
from google.colab import files

# Download final model
files.download('/content/ShifaMind_Capstone/03_Models/checkpoints/shifamind_model.pt')

# Download knowledge base
files.download('/content/ShifaMind_Capstone/03_Models/clinical_knowledge_base.json')
```

---

## 📊 Expected Results

After training completes, you should see:

### Performance Metrics
- **Macro F1**: ~0.80
- **AUROC**: ~0.91
- **Citation Completeness**: 100%

### Output Files
- `shifamind_model.pt` (final model, ~500 MB)
- `clinical_knowledge_base.json` (knowledge base, ~50 MB)
- `evaluation_report.json` (metrics)
- Various plots and visualizations

---

## 🆘 Need Help?

### Common Issues
1. **Path errors** → Verify Google Drive structure
2. **Memory errors** → Reduce batch size or use A100
3. **Timeout** → Keep browser active or use Colab Pro
4. **Import errors** → Re-run dependency installation cell

### Resources
- **README**: Detailed documentation
- **setup_paths.py**: Validate your data paths
- **requirements-colab.txt**: List of dependencies

### Support
- Check GitHub issues
- Review documentation in `docs/` folder
- Contact: Mohammed Sameer Syed

---

## 🎉 Success Checklist

After completing the notebook, you should have:

- [x] Google Drive mounted
- [x] Repository cloned to `/content/ShifaMind_Capstone`
- [x] All dependencies installed
- [x] Data paths validated
- [x] Clinical knowledge base generated
- [x] Model trained (3 stages)
- [x] Evaluation metrics computed
- [x] Model backed up to Google Drive
- [x] Ready to run inference!

---

**Enjoy using ShifaMind! 🏥🤖**

*Built with precision. Designed for transparency. Created for better healthcare.*
