# ShifaMind Training Pipeline

## Complete 3-Stage Training Process

```mermaid
graph TB
    subgraph "Data Preparation"
        A1[MIMIC-IV Discharge Notes]
        A2[UMLS Metathesaurus<br/>MRCONSO, MRDEF, MRSTY]
        A3[ICD-10-CM Codes]

        B1[Extract Target Diagnoses<br/>J189, I5023, A419, K8000]
        B2[Parse Medical Concepts<br/>Filter animal/veterinary terms]
        B3[Map ICD codes to descriptions]

        A1 --> B1
        A2 --> B2
        A3 --> B3

        C[Clinical Knowledge Base<br/>60 concepts per diagnosis]
        B1 --> C
        B2 --> C
        B3 --> C
    end

    subgraph "Stage 1: Diagnosis Head Training"
        D1[Bio_ClinicalBERT Base Model]
        D2[Add Diagnosis Classification Head<br/>4 output classes]
        D3[Freeze: Concept components]
        D4[Train: Diagnosis head only<br/>3 epochs, LR=2e-5]
        D5[Loss: Binary Cross-Entropy]
        D6[Checkpoint: stage1_diagnosis.pt]

        C --> D1
        D1 --> D2
        D2 --> D3
        D3 --> D4
        D4 --> D5
        D5 --> D6
    end

    subgraph "Stage 2: Concept Head Training"
        E1[Load Stage 1 Checkpoint]
        E2[Add Concept Classification Head<br/>60 concept outputs]
        E3[Add Cross-Attention Fusion Modules<br/>Layers 9 and 11]
        E4[Train: Concept head + fusion<br/>2 epochs, LR=2e-5]
        E5[Loss: 70% Concept BCE + 30% Confidence]
        E6[Checkpoint: stage2_concepts.pt]

        D6 --> E1
        E1 --> E2
        E2 --> E3
        E3 --> E4
        E4 --> E5
        E5 --> E6
    end

    subgraph "Stage 3: Joint Fine-tuning"
        F1[Load Stage 2 Checkpoint]
        F2[Add Bilinear Interaction Layer<br/>Diagnosis ⊗ Concepts]
        F3[Train: All components jointly<br/>3 epochs, LR=2e-5]
        F4[Multi-task Loss:<br/>50% Diagnosis + 25% Concepts + 25% Confidence]
        F5[Gradient Clipping: norm=1.0]
        F6[Final Model: shifamind_model.pt]

        E6 --> F1
        F1 --> F2
        F2 --> F3
        F3 --> F4
        F4 --> F5
        F5 --> F6
    end

    subgraph "Evaluation & Deployment"
        G1[Test Set Evaluation]
        G2[Compute Metrics:<br/>F1=0.801, AUROC=0.912]
        G3[Generate Visualizations]
        G4[Deploy Gradio Demo]

        F6 --> G1
        G1 --> G2
        G2 --> G3
        G3 --> G4
    end

    style D4 fill:#fff9c4
    style E4 fill:#fff9c4
    style F3 fill:#fff9c4
    style F6 fill:#c8e6c9
    style G2 fill:#c8e6c9
```

## Training Configuration

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Model | Bio_ClinicalBERT |
| Max Sequence Length | 384 tokens |
| Batch Size | 8 (training), 16 (eval) |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Warmup Ratio | 0.1 |
| Gradient Clipping | 1.0 |
| Random Seed | 42 |

### Stage-Specific Settings

| Stage | Epochs | Loss Function | Components Trained |
|-------|--------|---------------|-------------------|
| Stage 1 | 3 | BCE (diagnosis) | Diagnosis head only |
| Stage 2 | 2 | 70% Concept BCE + 30% Confidence | Concept head + fusion modules |
| Stage 3 | 3 | 50% Diagnosis + 25% Concepts + 25% Confidence | All components |

### Dataset Split

- **Training**: 70% (~2,100 samples per diagnosis)
- **Validation**: 15% (~450 samples per diagnosis)
- **Test**: 15% (~450 samples per diagnosis)

**Total**: ~3,000 samples per diagnosis × 4 diagnoses = 12,000 samples

### Compute Requirements

- **GPU**: NVIDIA T4 (16GB VRAM) or better
- **Training Time**: ~6 hours on T4, ~1 hour on A100
- **Storage**: ~2GB for checkpoints

## Inference Pipeline

```mermaid
graph LR
    A[Clinical Note] --> B[Tokenize]
    B --> C[BERT Encoding<br/>Layers 1-8]
    C --> D[Fusion Layer 9<br/>+ Medical Concepts]
    D --> E[Transformer Layer 10]
    E --> F[Fusion Layer 11<br/>+ Medical Concepts]
    F --> G[Transformer Layer 12]
    G --> H[CLS Representation]
    H --> I[Diagnosis Head]
    H --> J[Concept Head]
    I --> K[Bilinear Interaction]
    J --> K
    K --> L[Final Prediction]
    K --> M[Evidence Extraction]
    K --> N[Knowledge Retrieval]

    style L fill:#c8e6c9
    style M fill:#c8e6c9
    style N fill:#c8e6c9
```

---

**Render this file in any Markdown viewer that supports Mermaid diagrams**
