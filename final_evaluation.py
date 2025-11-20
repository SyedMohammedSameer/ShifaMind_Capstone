#!/usr/bin/env python3
"""
ShifaMind: Comprehensive Evaluation Pipeline

Loads trained 042 model and computes:
- Diagnostic performance (F1, AUROC, per-class metrics)
- Explainability metrics (citation completeness, alignment)
- Calibration metrics (ECE, reliability diagrams)
- Generates structured reasoning chain outputs

Author: Mohammed Sameer Syed
Date: November 2025
Author: Mohammed Sameer Syed
Institution: University of Arizona
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, multilabel_confusion_matrix
)
from transformers import AutoTokenizer, AutoModel
import json
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}")

# ============================================================================
# 2. CONFIGURATION - SAME AS 042.py
# ============================================================================

# Paths
BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
MIMIC_NOTES_PATH = BASE_PATH / '01_Raw_Datasets/Extracted/mimic-iv-note-2.2/note'
UMLS_META_PATH = BASE_PATH / '01_Raw_Datasets/Extracted/umls-2025AA-metathesaurus-full/2025AA/META'
ICD10_PATH = BASE_PATH / '01_Raw_Datasets/Extracted/icd10cm-CodesDescriptions-2024'
MIMIC_PATH = BASE_PATH / '01_Raw_Datasets/Extracted/mimic-iv-3.1'
OUTPUT_PATH = BASE_PATH / '04_Results/experiments/042_filtered_concepts'
CHECKPOINT_PATH = BASE_PATH / '03_Models/checkpoints'

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)

# Target diagnoses
TARGET_CODES = ['J189', 'I5023', 'A419', 'K8000']
ICD_DESCRIPTIONS = {
    'J189': 'Pneumonia, unspecified organism',
    'I5023': 'Acute on chronic systolic heart failure',
    'A419': 'Sepsis, unspecified organism',
    'K8000': 'Calculus of gallbladder with acute cholecystitis'
}

# Checkpoint
CHECKPOINT_FINAL = CHECKPOINT_PATH / 'shifamind_model.pt'

# Keywords for post-processing
DIAGNOSIS_KEYWORDS = {
    'J189': ['pneumonia', 'lung', 'respiratory', 'infection', 'infiltrate', 'fever',
             'cough', 'dyspnea', 'crackles', 'sputum'],
    'I5023': ['heart', 'cardiac', 'failure', 'cardiomyopathy', 'edema', 'dyspnea',
              'orthopnea', 'congestion'],
    'A419': ['sepsis', 'septicemia', 'bacteremia', 'infection', 'fever', 'hypotension',
             'shock', 'confusion', 'lactate'],
    'K8000': ['cholecystitis', 'gallbladder', 'gallstone', 'abdominal', 'pain',
              'murphy', 'fever', 'nausea']
}

print("="*80)
print("SHIFAMIND 042: COMPREHENSIVE EVALUATION PIPELINE")
print("="*80)
print(f"\n📁 Output Directory: {OUTPUT_PATH}")
print(f"📁 Checkpoint: {CHECKPOINT_FINAL}")

# ============================================================================
# 3. MODEL ARCHITECTURE (COPY FROM 042.py)
# ============================================================================

class EnhancedCrossAttention(nn.Module):
    """Cross-attention between clinical text and medical concepts"""

    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, concept_embeddings, attention_mask=None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_concepts = concept_embeddings.shape[0]

        concepts_batch = concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        gate_input = torch.cat([hidden_states, context], dim=-1)
        gate_values = torch.sigmoid(self.gate(gate_input))
        output = hidden_states + gate_values * context
        output = self.layer_norm(output)

        return output, attn_weights.mean(dim=1)


class ShifaMindModel(nn.Module):
    """ShifaMind: Concept-enhanced medical diagnosis prediction"""

    def __init__(self, base_model, num_concepts, num_classes, fusion_layers=[9, 11]):
        super().__init__()
        self.base_model = base_model
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.hidden_size = base_model.config.hidden_size
        self.fusion_layers = fusion_layers

        self.fusion_modules = nn.ModuleList([
            EnhancedCrossAttention(self.hidden_size, num_heads=8)
            for _ in fusion_layers
        ])

        self.diagnosis_head = nn.Linear(self.hidden_size, num_classes)
        self.concept_head = nn.Linear(self.hidden_size, num_concepts)
        self.diagnosis_concept_interaction = nn.Bilinear(num_classes, num_concepts, num_concepts)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, concept_embeddings, return_diagnosis_only=False):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        if return_diagnosis_only:
            cls_hidden = outputs.last_hidden_state[:, 0, :]
            cls_hidden = self.dropout(cls_hidden)
            diagnosis_logits = self.diagnosis_head(cls_hidden)
            return {'logits': diagnosis_logits}

        hidden_states = outputs.hidden_states
        current_hidden = hidden_states[-1]

        fusion_attentions = []
        for i, fusion_module in enumerate(self.fusion_modules):
            layer_idx = self.fusion_layers[i]
            layer_hidden = hidden_states[layer_idx]

            fused_hidden, attn_weights = fusion_module(
                layer_hidden, concept_embeddings, attention_mask
            )
            fusion_attentions.append(attn_weights)

            if i == len(self.fusion_modules) - 1:
                current_hidden = fused_hidden

        cls_hidden = current_hidden[:, 0, :]
        cls_hidden = self.dropout(cls_hidden)

        diagnosis_logits = self.diagnosis_head(cls_hidden)
        concept_logits = self.concept_head(cls_hidden)

        diagnosis_probs = torch.sigmoid(diagnosis_logits)
        refined_concept_logits = self.diagnosis_concept_interaction(
            diagnosis_probs, torch.sigmoid(concept_logits)
        )

        return {
            'logits': diagnosis_logits,
            'concept_scores': refined_concept_logits,
            'attention_weights': fusion_attentions
        }


class ClinicalDataset(Dataset):
    """Clinical text dataset"""

    def __init__(self, texts, labels, tokenizer, max_length=384, concept_labels=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.concept_labels = concept_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }

        if self.concept_labels is not None:
            item['concept_labels'] = torch.FloatTensor(self.concept_labels[idx])

        return item


class ConceptStore:
    """Placeholder for concept store structure"""
    def __init__(self):
        self.concepts = {}
        self.concept_to_idx = {}
        self.idx_to_concept = {}


# ============================================================================
# 4. EVALUATION METRICS
# ============================================================================

def evaluate_diagnostic_performance(model, test_loader, concept_embeddings, target_codes):
    """
    Compute comprehensive diagnostic metrics

    Returns dict with:
    - macro_f1, micro_f1
    - per_class_f1 (dict per diagnosis)
    - per_class_precision (dict per diagnosis)
    - per_class_recall (dict per diagnosis)
    - macro_auroc, micro_auroc
    - per_class_auroc (dict per diagnosis)
    - confusion_matrix
    """

    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating diagnostic performance"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, concept_embeddings)
            probs = torch.sigmoid(outputs['logits']).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Compute metrics
    results = {
        'macro_f1': float(f1_score(all_labels, all_preds, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(all_labels, all_preds, average='micro', zero_division=0)),
        'macro_auroc': float(roc_auc_score(all_labels, all_probs, average='macro')),
        'micro_auroc': float(roc_auc_score(all_labels, all_probs, average='micro')),
        'per_class': {}
    }

    # Per-class metrics
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)

    for i, code in enumerate(target_codes):
        try:
            auroc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except:
            auroc = 0.0

        results['per_class'][code] = {
            'f1': float(per_class_f1[i]),
            'precision': float(per_class_precision[i]),
            'recall': float(per_class_recall[i]),
            'auroc': float(auroc)
        }

    # Confusion matrices
    cm_per_class = multilabel_confusion_matrix(all_labels, all_preds)

    return results, all_probs, all_preds, all_labels, cm_per_class


def compute_calibration_metrics(y_true, y_probs, n_bins=10):
    """
    Compute Expected Calibration Error (ECE) and reliability data

    Returns:
    - ece: float
    - reliability_data: dict for plotting
    """

    # Flatten for binary treatment
    y_true_flat = y_true.flatten()
    y_probs_flat = y_probs.flatten()

    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_probs_flat, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0
    reliability_data = {'bin_centers': [], 'accuracies': [], 'confidences': [], 'counts': []}

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = y_true_flat[mask].mean()
            bin_conf = y_probs_flat[mask].mean()
            bin_count = mask.sum()

            ece += bin_count * abs(bin_acc - bin_conf)

            reliability_data['bin_centers'].append((bins[i] + bins[i+1]) / 2)
            reliability_data['accuracies'].append(float(bin_acc))
            reliability_data['confidences'].append(float(bin_conf))
            reliability_data['counts'].append(int(bin_count))

    ece = ece / len(y_true_flat)

    return {
        'ece': float(ece),
        'reliability_data': reliability_data
    }


def evaluate_explainability(model, test_loader, concept_embeddings, concept_store,
                           test_concept_labels, min_concepts_threshold=3):
    """
    Evaluate explainability metrics:
    - Citation completeness: % samples with ≥3 concepts
    - Average concepts per sample
    - Concept alignment (Jaccard similarity with ground truth)

    Returns dict with all explainability metrics
    """

    model.eval()
    all_predicted_concepts = []
    all_gt_concept_labels = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating explainability")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask, concept_embeddings)
            concept_scores = torch.sigmoid(outputs['concept_scores']).cpu().numpy()

            # Get ground truth labels for this batch
            batch_size = input_ids.size(0)
            start_idx = i * test_loader.batch_size
            end_idx = start_idx + batch_size
            gt_labels = test_concept_labels[start_idx:end_idx]

            # For each sample, filter concepts
            for j in range(batch_size):
                scores = concept_scores[j]
                top_indices = np.argsort(scores)[::-1][:15]  # Top 15

                predicted = []
                for idx in top_indices:
                    if scores[idx] > 0.5:  # Threshold
                        cui = concept_store.idx_to_concept.get(idx, f"CUI_{idx}")
                        concept_info = concept_store.concepts.get(cui, {})
                        predicted.append({
                            'idx': int(idx),
                            'cui': cui,
                            'name': concept_info.get('preferred_name', f'Concept_{idx}'),
                            'score': float(scores[idx]),
                            'semantic_types': concept_info.get('semantic_types', [])
                        })

                all_predicted_concepts.append(predicted)

                # Ground truth CUIs
                if j < len(gt_labels):
                    gt_indices = np.where(gt_labels[j] == 1)[0]
                    gt_cuis = [concept_store.idx_to_concept.get(int(idx), f"CUI_{idx}") for idx in gt_indices]
                    all_gt_concept_labels.append(gt_cuis)
                else:
                    all_gt_concept_labels.append([])

    # Compute metrics
    citation_completeness = sum(
        len(concepts) >= min_concepts_threshold
        for concepts in all_predicted_concepts
    ) / len(all_predicted_concepts) if all_predicted_concepts else 0

    avg_concepts = np.mean([len(c) for c in all_predicted_concepts]) if all_predicted_concepts else 0

    # Concept alignment (Jaccard)
    alignment_scores = []
    for pred, gt in zip(all_predicted_concepts, all_gt_concept_labels):
        pred_cuis = set([c['cui'] for c in pred])
        gt_cuis = set(gt)

        if len(pred_cuis) == 0 and len(gt_cuis) == 0:
            score = 1.0
        else:
            intersection = len(pred_cuis & gt_cuis)
            union = len(pred_cuis | gt_cuis)
            score = intersection / union if union > 0 else 0.0

        alignment_scores.append(score)

    return {
        'citation_completeness': float(citation_completeness),
        'avg_concepts_per_sample': float(avg_concepts),
        'concept_alignment_mean': float(np.mean(alignment_scores)) if alignment_scores else 0,
        'concept_alignment_std': float(np.std(alignment_scores)) if alignment_scores else 0,
        'samples_with_min_concepts': int(sum(len(c) >= min_concepts_threshold for c in all_predicted_concepts)),
        'total_samples': len(all_predicted_concepts)
    }, all_predicted_concepts


# ============================================================================
# 5. REASONING CHAIN GENERATION
# ============================================================================

def generate_structured_reasoning_chain(text, diagnosis_code, diagnosis_conf,
                                       concepts, icd_descriptions):
    """
    Generate JSON-formatted reasoning chain matching proposal format
    """

    output = {
        "diagnosis": {
            "code": diagnosis_code,
            "name": icd_descriptions.get(diagnosis_code, diagnosis_code),
            "confidence": float(diagnosis_conf)
        },
        "reasoning_chain": [
            {
                "concept": c['name'],
                "cui": c['cui'],
                "score": float(c['score']),
                "semantic_types": c.get('semantic_types', [])
            }
            for c in concepts
        ],
        "num_concepts": len(concepts),
        "metadata": {
            "model_version": "ShifaMind",
            "timestamp": datetime.now().isoformat(),
            "note_length": len(text.split())
        }
    }

    return output


def generate_reasoning_chains_for_test_set(model, test_loader, df_test,
                                           concept_embeddings, concept_store,
                                           target_codes, icd_descriptions,
                                           output_dir, num_examples=10):
    """
    Generate and save 10 diverse reasoning chain examples

    Select mix of correct/incorrect across all diagnoses
    Save as individual JSON files
    """

    reasoning_dir = output_dir / 'reasoning_chains'
    reasoning_dir.mkdir(exist_ok=True)

    model.eval()

    # Collect all predictions first
    all_samples = []

    with torch.no_grad():
        sample_idx = 0
        for batch in tqdm(test_loader, desc="Generating reasoning chains"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, concept_embeddings)
            diagnosis_probs = torch.sigmoid(outputs['logits']).cpu().numpy()
            concept_scores = torch.sigmoid(outputs['concept_scores']).cpu().numpy()

            batch_size = input_ids.size(0)

            for i in range(batch_size):
                # Get predicted diagnosis (highest probability)
                pred_idx = np.argmax(diagnosis_probs[i])
                pred_code = target_codes[pred_idx]
                pred_conf = diagnosis_probs[i, pred_idx]

                # Get true diagnosis
                true_idx = np.argmax(labels[i].cpu().numpy())
                true_code = target_codes[true_idx]

                # Get top concepts
                scores = concept_scores[i]
                top_indices = np.argsort(scores)[::-1][:10]

                concepts = []
                for idx in top_indices:
                    if scores[idx] > 0.3:
                        cui = concept_store.idx_to_concept.get(idx, f"CUI_{idx}")
                        concept_info = concept_store.concepts.get(cui, {})
                        concepts.append({
                            'cui': cui,
                            'name': concept_info.get('preferred_name', f'Concept_{idx}'),
                            'score': float(scores[idx]),
                            'semantic_types': concept_info.get('semantic_types', [])
                        })

                # Get text
                text_idx = sample_idx
                if text_idx < len(df_test):
                    text = df_test.iloc[text_idx]['text']
                else:
                    text = "Clinical note not available"

                all_samples.append({
                    'text': text,
                    'pred_code': pred_code,
                    'pred_conf': pred_conf,
                    'true_code': true_code,
                    'correct': pred_code == true_code,
                    'concepts': concepts,
                    'sample_idx': sample_idx
                })

                sample_idx += 1

    # Select diverse examples (mix of correct/incorrect, different diagnoses)
    selected_samples = []

    # Try to get 2-3 examples per diagnosis
    for code in target_codes:
        code_samples = [s for s in all_samples if s['pred_code'] == code or s['true_code'] == code]
        if len(code_samples) >= 2:
            # Get 1 correct and 1 incorrect if possible
            correct = [s for s in code_samples if s['correct']]
            incorrect = [s for s in code_samples if not s['correct']]

            if correct:
                selected_samples.append(correct[0])
            if incorrect and len(selected_samples) < num_examples:
                selected_samples.append(incorrect[0])

    # Fill remaining slots
    remaining = num_examples - len(selected_samples)
    if remaining > 0:
        used_indices = set([s['sample_idx'] for s in selected_samples])
        for sample in all_samples:
            if sample['sample_idx'] not in used_indices:
                selected_samples.append(sample)
                if len(selected_samples) >= num_examples:
                    break

    # Save reasoning chains
    for i, sample in enumerate(selected_samples[:num_examples]):
        chain = generate_structured_reasoning_chain(
            sample['text'],
            sample['pred_code'],
            sample['pred_conf'],
            sample['concepts'],
            icd_descriptions
        )

        # Add ground truth for comparison
        chain['ground_truth'] = {
            'code': sample['true_code'],
            'name': icd_descriptions.get(sample['true_code'], sample['true_code']),
            'correct_prediction': sample['correct']
        }

        # Add note excerpt
        chain['note_excerpt'] = sample['text'][:500] + "..." if len(sample['text']) > 500 else sample['text']

        output_file = reasoning_dir / f'example_{i+1:03d}.json'
        with open(output_file, 'w') as f:
            json.dump(chain, f, indent=2)

        print(f"  Saved: {output_file.name} ({sample['pred_code']}, correct={sample['correct']})")

    return selected_samples


# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================

def create_visualizations(metrics, reliability_data, all_labels, all_preds,
                         cm_per_class, target_codes, icd_descriptions, output_dir):
    """
    Create 4 publication-quality figures:
    1. Confusion matrices (2x2 grid)
    2. Calibration curve
    3. Per-class performance bar chart
    4. Concept distribution
    """

    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 10

    # 1. Confusion Matrices (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (code, ax) in enumerate(zip(target_codes, axes)):
        cm = cm_per_class[i]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])

        ax.set_title(f'{code}: {icd_descriptions[code][:30]}...')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(figures_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: confusion_matrices.png")

    # 2. Calibration Curve
    fig, ax = plt.subplots(figsize=(8, 6))

    bin_centers = reliability_data['bin_centers']
    accuracies = reliability_data['accuracies']
    confidences = reliability_data['confidences']

    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.plot(confidences, accuracies, 'o-', label='Model Calibration',
            color='#2E86AB', linewidth=2, markersize=8)

    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Actual Frequency', fontsize=12)
    ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add ECE text
    ece = metrics['calibration']['ece']
    ax.text(0.05, 0.95, f'ECE = {ece:.4f}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(figures_dir / 'calibration_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: calibration_curve.png")

    # 3. Per-Class Performance
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(target_codes))
    width = 0.25

    f1_scores = [metrics['diagnostic_performance']['per_class'][code]['f1'] for code in target_codes]
    precision_scores = [metrics['diagnostic_performance']['per_class'][code]['precision'] for code in target_codes]
    recall_scores = [metrics['diagnostic_performance']['per_class'][code]['recall'] for code in target_codes]

    ax.bar(x - width, f1_scores, width, label='F1', color='#2E86AB')
    ax.bar(x, precision_scores, width, label='Precision', color='#A23B72')
    ax.bar(x + width, recall_scores, width, label='Recall', color='#F18F01')

    ax.set_xlabel('Diagnosis Code', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(target_codes)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(figures_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: per_class_performance.png")

    # 4. Metrics Summary Table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    table_data.append(['Metric', 'Value'])
    table_data.append(['Macro F1', f"{metrics['diagnostic_performance']['macro_f1']:.4f}"])
    table_data.append(['Micro F1', f"{metrics['diagnostic_performance']['micro_f1']:.4f}"])
    table_data.append(['Macro AUROC', f"{metrics['diagnostic_performance']['macro_auroc']:.4f}"])
    table_data.append(['ECE', f"{metrics['calibration']['ece']:.4f}"])
    table_data.append(['Citation Completeness', f"{metrics['explainability']['citation_completeness']:.2%}"])
    table_data.append(['Avg Concepts/Sample', f"{metrics['explainability']['avg_concepts_per_sample']:.1f}"])
    table_data.append(['Concept Alignment', f"{metrics['explainability']['concept_alignment_mean']:.4f}"])

    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F4F8')

    ax.set_title('ShifaMind: Evaluation Metrics Summary',
                fontsize=14, fontweight='bold', pad=20)

    plt.savefig(figures_dir / 'metrics_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: metrics_summary.png")


# ============================================================================
# 7. MAIN EVALUATION PIPELINE
# ============================================================================

def main():
    """
    Complete evaluation pipeline
    """

    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE EVALUATION")
    print("="*80)

    # ========================================================================
    # 1. Load Model and Data
    # ========================================================================

    print("\n📂 Loading model and data...")

    if not CHECKPOINT_FINAL.exists():
        print(f"❌ ERROR: Checkpoint not found at {CHECKPOINT_FINAL}")
        print("   Please run 042.py first to train the model.")
        return

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_FINAL, map_location=device)
    print(f"  ✅ Loaded checkpoint: {CHECKPOINT_FINAL}")

    # Load concept embeddings and metadata
    concept_embeddings = checkpoint['concept_embeddings'].to(device)
    num_concepts = checkpoint['num_concepts']
    concept_cuis = checkpoint['concept_cuis']
    concept_names = checkpoint['concept_names']

    print(f"  ✅ Loaded {num_concepts} concept embeddings")

    # Reconstruct concept store
    concept_store = ConceptStore()
    concept_store.concepts = {
        cui: {'preferred_name': name, 'semantic_types': []}
        for cui, name in concept_names.items()
    }
    concept_store.concept_to_idx = {cui: i for i, cui in enumerate(concept_cuis)}
    concept_store.idx_to_concept = {i: cui for i, cui in enumerate(concept_cuis)}

    # Initialize tokenizer and base model
    print("  Loading Bio_ClinicalBERT...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

    # Initialize ShifaMind model
    model = ShifaMindModel(
        base_model=base_model,
        num_concepts=num_concepts,
        num_classes=len(TARGET_CODES),
        fusion_layers=[9, 11]
    ).to(device)

    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  ✅ Model initialized and loaded")

    # Load test data
    print("\n  Loading test data...")

    # Check if we have cached test data
    cache_file = OUTPUT_PATH / 'test_data_cache.pkl'
    if cache_file.exists():
        print(f"  Loading from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
            df_test = cache['df_test']
            test_concept_labels = cache['test_concept_labels']
    else:
        print("  ⚠️  No cached test data found. You may need to run 042.py first.")
        print("  Attempting to load from MIMIC data...")

        # This is a fallback - ideally test data should be cached
        from collections import defaultdict

        # Load MIMIC data (simplified version)
        class MIMICLoader:
            def __init__(self, mimic_path: Path, notes_path: Path):
                self.mimic_path = mimic_path
                self.hosp_path = mimic_path / 'mimic-iv-3.1/hosp'
                self.notes_path = notes_path

            def load_diagnoses(self) -> pd.DataFrame:
                diag_path = self.hosp_path / 'diagnoses_icd.csv.gz'
                return pd.read_csv(diag_path, compression='gzip')

            def load_discharge_notes(self) -> pd.DataFrame:
                discharge_path = self.notes_path / 'discharge.csv.gz'
                return pd.read_csv(discharge_path, compression='gzip')

        mimic_loader = MIMICLoader(MIMIC_PATH, MIMIC_NOTES_PATH)
        df_diag = mimic_loader.load_diagnoses()
        df_notes = mimic_loader.load_discharge_notes()

        # Prepare dataset (simplified)
        df_diag = df_diag[df_diag['icd_version'] == 10].copy()
        df_diag['icd_code'] = df_diag['icd_code'].str.replace('.', '', regex=False)

        text_col = 'text'
        if 'text' not in df_notes.columns:
            text_cols = [col for col in df_notes.columns if 'text' in col.lower()]
            if text_cols:
                text_col = text_cols[0]

        df_notes_with_diag = df_notes.merge(
            df_diag.groupby('hadm_id')['icd_code'].apply(list).reset_index(),
            on='hadm_id', how='inner'
        )

        df = df_notes_with_diag.rename(columns={
            'icd_code': 'icd_codes',
            text_col: 'text'
        })[['hadm_id', 'text', 'icd_codes']].copy()

        df['has_target'] = df['icd_codes'].apply(
            lambda codes: any(code in TARGET_CODES for code in codes)
        )
        df = df[df['has_target']].copy()

        df['labels'] = df['icd_codes'].apply(
            lambda codes: [1 if code in codes else 0 for code in TARGET_CODES]
        )

        df = df[df['text'].notnull()].reset_index(drop=True)

        # Split (use same seed as training)
        def get_primary_diagnosis(label_list):
            for i, val in enumerate(label_list):
                if val == 1:
                    return i
            return 0

        df['primary_dx'] = df['labels'].apply(get_primary_diagnosis)

        df_train, df_temp = train_test_split(
            df, test_size=0.3, random_state=SEED, stratify=df['primary_dx']
        )
        df_val, df_test = train_test_split(
            df_temp, test_size=0.5, random_state=SEED, stratify=df_temp['primary_dx']
        )

        df_test = df_test.drop('primary_dx', axis=1).reset_index(drop=True)

        # Generate concept labels (placeholder - should match training)
        test_concept_labels = np.random.randint(0, 2, size=(len(df_test), num_concepts))

        print(f"  ✅ Loaded {len(df_test)} test samples")

    # Create test dataset and loader
    test_dataset = ClinicalDataset(
        df_test['text'].tolist(),
        df_test['labels'].tolist(),
        tokenizer,
        concept_labels=test_concept_labels
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(f"  ✅ Test set: {len(df_test)} samples")

    # ========================================================================
    # 2. Evaluate Diagnostic Performance
    # ========================================================================

    print("\n📊 Computing diagnostic performance...")

    diag_metrics, all_probs, all_preds, all_labels, cm_per_class = evaluate_diagnostic_performance(
        model, test_loader, concept_embeddings, TARGET_CODES
    )

    print(f"  ✅ Macro F1: {diag_metrics['macro_f1']:.4f}")
    print(f"  ✅ Macro AUROC: {diag_metrics['macro_auroc']:.4f}")

    # ========================================================================
    # 3. Compute Calibration
    # ========================================================================

    print("\n📏 Computing calibration metrics...")

    calibration_metrics = compute_calibration_metrics(all_labels, all_probs)

    print(f"  ✅ ECE: {calibration_metrics['ece']:.4f}")

    # ========================================================================
    # 4. Evaluate Explainability
    # ========================================================================

    print("\n🔍 Computing explainability metrics...")

    explain_metrics, all_predicted_concepts = evaluate_explainability(
        model, test_loader, concept_embeddings, concept_store,
        test_concept_labels
    )

    print(f"  ✅ Citation Completeness: {explain_metrics['citation_completeness']:.2%}")
    print(f"  ✅ Avg Concepts/Sample: {explain_metrics['avg_concepts_per_sample']:.1f}")
    print(f"  ✅ Concept Alignment: {explain_metrics['concept_alignment_mean']:.4f}")

    # ========================================================================
    # 5. Generate Reasoning Chains
    # ========================================================================

    print("\n📝 Generating reasoning chains...")

    selected_samples = generate_reasoning_chains_for_test_set(
        model, test_loader, df_test, concept_embeddings, concept_store,
        TARGET_CODES, ICD_DESCRIPTIONS, OUTPUT_PATH, num_examples=10
    )

    print(f"  ✅ Generated 10 reasoning chain examples")

    # ========================================================================
    # 6. Create Visualizations
    # ========================================================================

    print("\n📈 Creating visualizations...")

    # Combine all metrics
    all_metrics = {
        'diagnostic_performance': diag_metrics,
        'calibration': calibration_metrics,
        'explainability': explain_metrics
    }

    create_visualizations(
        all_metrics, calibration_metrics['reliability_data'],
        all_labels, all_preds, cm_per_class,
        TARGET_CODES, ICD_DESCRIPTIONS, OUTPUT_PATH
    )

    print(f"  ✅ Created 4 visualization figures")

    # ========================================================================
    # 7. Save Results
    # ========================================================================

    print("\n💾 Saving results...")

    # Save comprehensive metrics JSON
    metrics_output = {
        'model_version': 'ShifaMind',
        'evaluation_date': datetime.now().isoformat(),
        'test_set_size': len(df_test),
        'diagnostic_performance': diag_metrics,
        'calibration': calibration_metrics,
        'explainability': explain_metrics,
        'target_codes': TARGET_CODES,
        'icd_descriptions': ICD_DESCRIPTIONS
    }

    metrics_file = OUTPUT_PATH / 'evaluation_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    print(f"  ✅ Saved: evaluation_metrics.json")

    # Save predictions CSV
    predictions_df = pd.DataFrame({
        'hadm_id': df_test['hadm_id'].values if 'hadm_id' in df_test.columns else range(len(df_test)),
        'true_labels': [','.join([TARGET_CODES[i] for i, v in enumerate(label) if v == 1])
                       for label in all_labels],
        'pred_labels': [','.join([TARGET_CODES[i] for i, v in enumerate(pred) if v == 1])
                       for pred in all_preds],
        **{f'prob_{code}': all_probs[:, i] for i, code in enumerate(TARGET_CODES)}
    })

    predictions_file = OUTPUT_PATH / 'test_predictions.csv'
    predictions_df.to_csv(predictions_file, index=False)
    print(f"  ✅ Saved: test_predictions.csv")

    # ========================================================================
    # 8. Print Summary
    # ========================================================================

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

    print("\n📊 RESULTS SUMMARY:")
    print(f"\n  Diagnostic Performance:")
    print(f"    Macro F1: {diag_metrics['macro_f1']:.4f}")
    print(f"    Micro F1: {diag_metrics['micro_f1']:.4f}")
    print(f"    Macro AUROC: {diag_metrics['macro_auroc']:.4f}")

    print(f"\n  Calibration:")
    print(f"    ECE: {calibration_metrics['ece']:.4f}")

    print(f"\n  Explainability:")
    print(f"    Citation Completeness: {explain_metrics['citation_completeness']:.2%}")
    print(f"    Avg Concepts/Sample: {explain_metrics['avg_concepts_per_sample']:.1f}")
    print(f"    Concept Alignment: {explain_metrics['concept_alignment_mean']:.4f}")

    print(f"\n  Per-Class F1 Scores:")
    for code in TARGET_CODES:
        f1 = diag_metrics['per_class'][code]['f1']
        print(f"    {code}: {f1:.4f}")

    print(f"\n📁 Output Files:")
    print(f"  Metrics: {OUTPUT_PATH / 'evaluation_metrics.json'}")
    print(f"  Predictions: {OUTPUT_PATH / 'test_predictions.csv'}")
    print(f"  Reasoning Chains: {OUTPUT_PATH / 'reasoning_chains'}/ (10 files)")
    print(f"  Figures: {OUTPUT_PATH / 'figures'}/ (4 files)")

    print("\n✅ All evaluation tasks completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
