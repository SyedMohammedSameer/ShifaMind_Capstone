# ShifaMind: Enforced Explainability in Clinical AI

![Status](https://img.shields.io/badge/status-production-green)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-Private-red)

**ShifaMind** is a deep learning system for clinical diagnosis prediction with enforced explainability. It combines transformer-based language models with medical ontologies to provide accurate, interpretable diagnoses from clinical notes.

**Author**: Mohammed Sameer Syed
**Institution**: University of Arizona
**Program**: M.S. in Artificial Intelligence
**Project**: Capstone (2025)

---

## Table of Contents

- [Overview](#overview)
- [Key Innovations](#key-innovations)
- [Performance](#performance)
- [Quick Start](#quick-start)
- [🚀 Google Colab Setup](#-google-colab-setup-easiest-method)
- [Installation (Local)](#installation-local-setup)
- [Usage](#usage)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)

---

## Overview

ShifaMind addresses a critical challenge in clinical AI: the lack of explainability. While modern deep learning models can achieve high diagnostic accuracy, they often function as "black boxes," making it difficult for clinicians to trust and validate their predictions.

**ShifaMind solves this by:**

1. **Deep Ontology Fusion**: Integrating medical concepts from UMLS/ICD-10 directly into transformer layers
2. **Forced Citation Mechanism**: Requiring the model to cite specific medical concepts for every prediction
3. **Evidence Extraction**: Automatically identifying text spans that support each cited concept
4. **Clinical Knowledge Retrieval**: Providing relevant medical knowledge for each prediction

The result is a system that not only predicts diagnoses accurately but also explains *why* and *how* it arrived at each conclusion.

---

## Key Innovations

### 1. Deep Ontology Fusion at Every Transformer Layer

Unlike traditional approaches that add medical knowledge as a post-processing step, ShifaMind fuses medical concepts **directly into the transformer's attention mechanism** at layers 9 and 11.

```
Clinical Text → BERT Layers 1-8 → [Concept Fusion @ Layer 9] →
                → [Concept Fusion @ Layer 11] → Diagnosis + Concepts
```

This deep integration allows the model to reason about medical concepts throughout the encoding process, not just at the output.

### 2. Attention-Based Evidence Extraction

ShifaMind uses cross-attention weights to identify which parts of the clinical note contributed to each predicted concept. This provides:

- **Text spans** that justify each medical concept
- **Attention scores** showing the strength of evidence
- **Traceability** from raw text → concepts → diagnosis

### 3. Automated Clinical Knowledge Base

The system automatically generates a structured knowledge base from:

- **UMLS (Unified Medical Language System)**: 200,000+ medical concepts
- **ICD-10-CM**: Official diagnosis codes and descriptions
- **Filtered Content**: Animal/veterinary concepts automatically removed

No manual curation required—the knowledge base is built from authoritative medical sources.

### 4. Filtered Human-Only Medicine

All predictions are filtered to remove veterinary and animal medical concepts, ensuring that only human clinical medicine is presented to users.

---

## Performance

ShifaMind achieves state-of-the-art performance on MIMIC-IV diagnostic prediction:

| Metric | Score | Improvement |
|--------|-------|-------------|
| **Macro F1** | 0.8010 | +8.6% over baseline |
| **AUROC** | 0.9122 | +5.3% over baseline |
| **Citation Completeness** | 100% | N/A (unique to ShifaMind) |
| **Concept Alignment** | 0.7845 | N/A (unique to ShifaMind) |

**Supported Diagnoses:**

- **J189**: Pneumonia, unspecified organism
- **I5023**: Acute on chronic systolic heart failure
- **A419**: Sepsis, unspecified organism
- **K8000**: Calculus of gallbladder with acute cholecystitis

---

## Quick Start

### Inference (Predict on New Text)

```python
from final_inference import ShifaMindPredictor

# Initialize predictor
predictor = ShifaMindPredictor()

# Predict on clinical note
result = predictor.predict("""
72-year-old male presents with fever (38.9°C), productive cough,
and shortness of breath. Chest X-ray shows right lower lobe infiltrate.
""")

# Display results
print(f"Diagnosis: {result['diagnosis']['name']}")
print(f"Confidence: {result['diagnosis']['confidence']:.1%}")
print(f"Top Concepts: {[c['name'] for c in result['concepts'][:5]]}")
```

### CLI Usage

```bash
# Predict from text
python final_inference.py --text "Patient with fever and cough..." --format detailed

# Predict from file
python final_inference.py --file clinical_note.txt --save result.json
```

---

## 🚀 Google Colab Setup (Easiest Method)

**Want to run ShifaMind with ZERO local setup?** Use our pre-configured Google Colab notebook!

### One-Click Colab Setup

1. **Upload your data to Google Drive** (if not already there):
   ```
   My Drive/
   └── ShifaMind/
       └── 01_Raw_Datasets/Extracted/
           ├── umls-2025AA-metathesaurus-full/
           ├── icd10cm-CodesDescriptions-2024/
           ├── mimic-iv-3.1/
           └── mimic-iv-note-2.2/
   ```

2. **Open the Colab notebook**:
   - Upload `colab_setup.ipynb` to Google Colab, or
   - Click: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SyedMohammedSameer/ShifaMind_Capstone/blob/main/colab_setup.ipynb)

3. **Run all cells** - that's it! The notebook will:
   - Mount your Google Drive
   - Clone the repository
   - Install dependencies (only what Colab doesn't have)
   - Configure paths automatically
   - Run the complete pipeline

### What You Get

✅ **No local installation required**
✅ **Free GPU access** (T4, or A100 with Colab Pro)
✅ **All dependencies pre-installed** (PyTorch, CUDA, etc.)
✅ **Automatic path configuration**
✅ **Interactive demo interface**
✅ **Step-by-step guidance**

### Custom Paths on Colab

If your data is at a different Google Drive location, use our path helper:

```python
# In Colab
!python setup_paths.py --base-path /content/drive/MyDrive/YourFolder/ShifaMind
```

Or set the environment variable manually:

```python
import os
os.environ['SHIFAMIND_BASE_PATH'] = '/content/drive/MyDrive/YourFolder/ShifaMind'
```

### Colab Tips

- **GPU**: Enable GPU via Runtime → Change runtime type → Select T4 GPU
- **Timeout**: Keep browser tab active during training, or use Colab Pro
- **Storage**: Training generates ~2GB of model checkpoints
- **Speed**: Full training takes ~6 hours on T4, ~1 hour on A100

---

## Installation (Local Setup)

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM
- Access to:
  - MIMIC-IV dataset (for training)
  - UMLS Metathesaurus (for knowledge base)
  - ICD-10-CM codes

### Step 1: Clone Repository

```bash
git clone https://github.com/SyedMohammedSameer/ShifaMind.git
cd ShifaMind
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download Medical Data

1. **UMLS Metathesaurus** (requires free UMLS account):
   - Register at https://uts.nlm.nih.gov/
   - Download UMLS 2025AA Full Release
   - Extract to `01_Raw_Datasets/Extracted/umls-2025AA-metathesaurus-full/`

2. **MIMIC-IV** (requires credentialed access):
   - Complete CITI training at https://physionet.org/
   - Download MIMIC-IV v3.1
   - Extract to `01_Raw_Datasets/Extracted/mimic-iv-3.1/`

3. **ICD-10-CM Codes** (public):
   - Download from https://www.cms.gov/medicare/coding-billing/icd-10-codes
   - Extract to `01_Raw_Datasets/Extracted/icd10cm-CodesDescriptions-2024/`

### Step 4: Configuration

Edit `config.py` to set your data paths:

```python
BASE_PATH = Path('/your/path/to/ShifaMind')
```

Or set an environment variable:

```bash
export SHIFAMIND_BASE_PATH=/your/path/to/ShifaMind
```

---

## Usage

### 1. Generate Knowledge Base

```bash
python final_knowledge_base_generator.py
```

This creates `clinical_knowledge_base.json` from UMLS and ICD-10 data (~5-10 minutes).

### 2. Train Model

```bash
python final_model_training.py
```

Training consists of 3 stages:
- **Stage 1**: Diagnosis head (3 epochs, ~2 hours)
- **Stage 2**: Concept head (2 epochs, ~1.5 hours)
- **Stage 3**: Joint fine-tuning (3 epochs, ~2 hours)

**Total time**: ~6 hours on V100 GPU

### 3. Evaluate Model

```bash
python final_evaluation.py
```

Generates comprehensive metrics and visualizations:
- Diagnostic performance (F1, AUROC, per-class metrics)
- Calibration metrics (ECE, reliability diagrams)
- Explainability metrics (citation completeness, concept alignment)
- Reasoning chain examples

### 4. Launch Demo

```bash
python final_demo.py
```

Launches interactive Gradio interface at `http://localhost:7860` with:
- Real-time diagnosis prediction
- Evidence span extraction
- Clinical knowledge retrieval
- JSON output for integration

### 5. Use as Library

```python
from final_inference import ShifaMindPredictor

predictor = ShifaMindPredictor()

# Single prediction
result = predictor.predict(clinical_note_text)

# Batch prediction
results = predictor.predict_batch(list_of_texts)
```

---

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                      Clinical Note (Text)                       │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Bio_ClinicalBERT Tokenizer                     │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           Transformer Layers 1-8 (Standard BERT)                │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 9: Cross-Attention Fusion with Medical Concepts          │
│  (Attention between text tokens and UMLS concept embeddings)    │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Transformer Layer 10                          │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 11: Cross-Attention Fusion with Medical Concepts         │
│  (Second fusion point for refined concept integration)          │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Transformer Layer 12                        │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                   ┌─────────┴─────────┐
                   │                   │
                   ▼                   ▼
         ┌──────────────────┐  ┌──────────────────┐
         │  Diagnosis Head  │  │   Concept Head   │
         │  (4 diagnoses)   │  │  (60 concepts)   │
         └──────────────────┘  └──────────────────┘
                   │                   │
                   │        ┌──────────┘
                   │        │
                   ▼        ▼
         ┌─────────────────────────────┐
         │ Diagnosis-Concept Bilinear  │
         │     (Alignment Layer)       │
         └─────────────────────────────┘
                   │
                   ▼
         ┌─────────────────────────────┐
         │  Final Prediction Output:   │
         │  - Diagnosis (with conf.)   │
         │  - Concepts (filtered)      │
         │  - Evidence spans           │
         │  - Knowledge retrieval      │
         └─────────────────────────────┘
```

### Key Components

1. **Base Model**: Bio_ClinicalBERT (110M parameters)
2. **Concept Embeddings**: Pre-encoded UMLS concepts (60-150 per diagnosis)
3. **Fusion Modules**: Enhanced cross-attention at layers 9 and 11
4. **Output Heads**:
   - Diagnosis head (4-class multi-label)
   - Concept head (60-concept multi-label)
5. **Alignment Layer**: Bilinear interaction between diagnoses and concepts

**Total Parameters**: ~125M

---

## Project Structure

```
ShifaMind/
├── config.py                              # Centralized configuration
├── setup_paths.py                         # Path configuration helper
├── final_concept_filter.py                 # Animal concept filtering
├── final_knowledge_base_generator.py       # KB creation from UMLS/ICD-10
├── final_model_training.py                 # Complete training pipeline
├── final_evaluation.py                     # Comprehensive evaluation
├── final_inference.py                      # Standalone inference
├── final_demo.py                          # Interactive Gradio demo
├── colab_setup.ipynb                      # 🚀 ONE-CLICK Colab setup
├── requirements.txt                        # Python dependencies (full)
├── requirements-colab.txt                  # Python dependencies (Colab only)
├── README.md                              # This file
├── LICENSE                                # License information
├── docs/
│   ├── ARCHITECTURE.md                    # Technical architecture details
│   ├── SETUP.md                          # Detailed setup instructions
│   └── USAGE.md                          # Usage examples and API
└── examples/
    └── sample_clinical_notes.json        # Example clinical notes
```

---

## Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Detailed technical architecture
- **[SETUP.md](docs/SETUP.md)**: Step-by-step setup guide
- **[USAGE.md](docs/USAGE.md)**: Comprehensive usage examples
- **[API Reference](docs/API.md)**: Python API documentation (coming soon)

---

## Citation

If you use ShifaMind in your research, please cite:

```bibtex
@mastersthesis{syed2025shifamind,
  author = {Syed, Mohammed Sameer},
  title = {ShifaMind: Enforced Explainability in Clinical AI through Deep Ontology Fusion},
  school = {University of Arizona},
  year = {2025},
  type = {M.S. Capstone Project},
  note = {M.S. in Artificial Intelligence}
}
```

**Alternative format:**

Syed, M. S. (2025). *ShifaMind: Enforced Explainability in Clinical AI through Deep Ontology Fusion*. M.S. Capstone Project, University of Arizona.

---

## License

This project is currently **private** and not licensed for public use or redistribution.

**Copyright © 2025 Mohammed Sameer Syed. All rights reserved.**

For licensing inquiries, please contact the author.

---

## Acknowledgments

- **University of Arizona** - M.S. in Artificial Intelligence Program
- **MIMIC-IV** - Johnson, A., Bulgarelli, L., Pollard, T., et al. (2023)
- **UMLS** - National Library of Medicine, NIH
- **Bio_ClinicalBERT** - Alsentzer, E., et al. (2019)

---

## Technical Support

For technical issues or questions:

1. Check the [documentation](docs/)
2. Review [example notebooks](examples/)
3. Contact: Mohammed Sameer Syed ([GitHub](https://github.com/SyedMohammedSameer))

---

## Roadmap

- [x] Core model training and evaluation
- [x] Interactive demo interface
- [x] Standalone inference module
- [ ] REST API for deployment
- [ ] Docker containerization
- [ ] Clinical trial validation
- [ ] Extended diagnosis support (10+ diagnoses)
- [ ] Multi-language support

---

**Built with precision. Designed for transparency. Created for better healthcare.**
