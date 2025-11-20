"""
ShifaMind: Complete Model Training Pipeline

This script trains the ShifaMind model with deep ontology fusion and forced
citation mechanisms through a three-stage training process:

Stage 1: Diagnosis Head Training (3 epochs)
    - Trains only the diagnosis prediction head
    - Freezes concept-related components
    - Optimizes for diagnostic accuracy

Stage 2: Concept Head Training (2 epochs)
    - Trains concept prediction and alignment
    - Uses whitelist labels for supervision
    - Optimizes for concept precision

Stage 3: Joint Fine-tuning (3 epochs)
    - Fine-tunes all components together
    - Enforces diagnosis-concept alignment
    - Produces final production model

Author: Mohammed Sameer Syed
Institution: University of Arizona
Project: M.S. in Artificial Intelligence Capstone
Date: November 2025
"""

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
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import json
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
import pickle
import math
import argparse
import logging
from datetime import datetime

# Import configuration
from config import (
    BASE_PATH, MIMIC_NOTES_PATH, UMLS_PATH, ICD10_PATH, MIMIC_PATH,
    MODEL_PATH, CHECKPOINT_PATH, RESULTS_PATH,
    TARGET_CODES, ICD_DESCRIPTIONS, REQUIRED_MEDICAL_TERMS, DIAGNOSIS_KEYWORDS,
    BASE_MODEL_NAME, FUSION_LAYERS, HIDDEN_SIZE, DROPOUT,
    BATCH_SIZE, EVAL_BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    NUM_EPOCHS_STAGE1, NUM_EPOCHS_STAGE2, NUM_EPOCHS_STAGE3,
    MAX_SEQUENCE_LENGTH, MAX_SAMPLES_PER_DIAGNOSIS,
    RANDOM_SEED, GRADIENT_CLIP_NORM,
    CHECKPOINT_STAGE1, CHECKPOINT_STAGE2, CHECKPOINT_FINAL,
    LOSS_WEIGHT_DIAGNOSIS, LOSS_WEIGHT_CONCEPT, LOSS_WEIGHT_CONFIDENCE,
    CONCEPT_LOSS_WEIGHT, CONFIDENCE_LOSS_WEIGHT,
    TOP_N_CONCEPTS_PER_DIAGNOSIS, TRUSTED_UMLS_SOURCES,
    get_device
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Device configuration
device = get_device()
logger.info(f"🖥️  Device: {device}")

# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_data_structure():
    """Validate all required data is present"""

    logger.info("="*70)
    logger.info("DATA VALIDATION")
    logger.info("="*70)

    checks = {
        'MIMIC Notes': MIMIC_NOTES_PATH / 'discharge.csv.gz',
        'UMLS MRCONSO': UMLS_PATH / 'MRCONSO.RRF',
        'UMLS MRSTY': UMLS_PATH / 'MRSTY.RRF',
        'MIMIC Diagnoses': MIMIC_PATH / 'mimic-iv-3.1/hosp/diagnoses_icd.csv.gz',
        'Output Directory': RESULTS_PATH,
        'Checkpoint Directory': CHECKPOINT_PATH
    }

    all_valid = True
    for name, path in checks.items():
        if path.exists():
            if path.is_file():
                size_mb = path.stat().st_size / (1024**2)
                logger.info(f"   ✅ {name}: {size_mb:.1f} MB")
            else:
                logger.info(f"   ✅ {name}: exists")
        else:
            logger.error(f"   ❌ {name}: NOT FOUND at {path}")
            all_valid = False

    if not all_valid:
        raise FileNotFoundError("Missing required data files!")

    logger.info(f"✅ All data validation checks passed")
    return True

# ============================================================================
# TARGETED UMLS LOADER
# ============================================================================

class TargetedUMLSLoader:
    """Load ONLY the medical concepts we actually need"""

    def __init__(self, umls_path: Path):
        self.umls_path = umls_path
        self.mrconso_path = umls_path / 'MRCONSO.RRF'
        self.mrsty_path = umls_path / 'MRSTY.RRF'
        self.mrdef_path = umls_path / 'MRDEF.RRF'

    def load_specific_concepts(self, required_terms: Dict[str, List[str]]):
        """Search UMLS for specific medical terms"""

        logger.info("="*70)
        logger.info("TARGETED UMLS CONCEPT LOADING")
        logger.info("="*70)

        # Flatten all required terms
        all_terms_flat = []
        for dx_code, terms_list in required_terms.items():
            all_terms_flat.extend(terms_list)

        # Create normalized search terms
        search_terms = set([t.lower().strip() for t in all_terms_flat])

        logger.info(f"🎯 Searching UMLS for {len(search_terms)} specific medical terms:")
        for dx_code, terms in required_terms.items():
            logger.info(f"   {dx_code}: {len(terms)} terms")

        # Search MRCONSO
        found_concepts = {}
        term_to_cuis = defaultdict(list)

        logger.info("📖 Scanning MRCONSO (17M+ entries)...")
        logger.info("   This will take ~30-60 seconds...")

        with open(self.mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="  Searching"):
                fields = line.strip().split('|')
                if len(fields) < 15:
                    continue

                cui, lang, sab, code, term = fields[0], fields[1], fields[11], fields[13], fields[14]

                # Only English from trusted sources
                if lang != 'ENG':
                    continue
                if sab not in TRUSTED_UMLS_SOURCES:
                    continue

                term_lower = term.lower().strip()

                # Check if this UMLS term matches any of our search terms
                matched_search_term = None
                for search_term in search_terms:
                    # Exact match or contains match
                    if search_term == term_lower or search_term in term_lower or term_lower in search_term:
                        matched_search_term = search_term
                        break

                if matched_search_term:
                    # Store concept
                    if cui not in found_concepts:
                        found_concepts[cui] = {
                            'cui': cui,
                            'preferred_name': term,
                            'terms': [term],
                            'sources': {sab: [code]},
                            'semantic_types': [],
                            'definition': ''
                        }
                    else:
                        # Add synonym
                        if term not in found_concepts[cui]['terms']:
                            found_concepts[cui]['terms'].append(term)
                        if sab not in found_concepts[cui]['sources']:
                            found_concepts[cui]['sources'][sab] = []
                        if code and code not in found_concepts[cui]['sources'][sab]:
                            found_concepts[cui]['sources'][sab].append(code)

                    # Map search term to CUI
                    if cui not in term_to_cuis[matched_search_term]:
                        term_to_cuis[matched_search_term].append(cui)

        logger.info(f"  ✅ Found {len(found_concepts)} unique concepts")

        # Show coverage per diagnosis
        logger.info("  📊 Coverage per diagnosis:")
        dx_coverage = {}
        for dx_code, terms_list in required_terms.items():
            found_for_dx = set()
            for term in terms_list:
                term_lower = term.lower().strip()
                if term_lower in term_to_cuis:
                    found_for_dx.update(term_to_cuis[term_lower])
            dx_coverage[dx_code] = len(found_for_dx)
            logger.info(f"    {dx_code}: {len(found_for_dx)} concepts ({len(terms_list)} terms searched)")

        # Load semantic types
        logger.info("📋 Loading semantic types...")
        cui_to_types = self._load_semantic_types(set(found_concepts.keys()))

        for cui, types in cui_to_types.items():
            if cui in found_concepts:
                found_concepts[cui]['semantic_types'] = types

        logger.info(f"  ✅ Added semantic types for {len(cui_to_types)} concepts")

        # Load definitions
        logger.info("📖 Loading definitions...")
        definitions_added = self._load_definitions(found_concepts)
        logger.info(f"  ✅ Added {definitions_added} definitions")

        return found_concepts, term_to_cuis, dx_coverage

    def _load_semantic_types(self, target_cuis: Set[str]) -> Dict[str, List[str]]:
        """Load semantic types only for found CUIs"""
        cui_to_types = defaultdict(list)

        with open(self.mrsty_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split('|')
                if len(fields) >= 2:
                    cui = fields[0]
                    if cui in target_cuis:
                        cui_to_types[cui].append(fields[1])

        return cui_to_types

    def _load_definitions(self, concepts: Dict) -> int:
        """Load definitions for found concepts"""
        if not self.mrdef_path.exists():
            return 0

        definitions_added = 0

        with open(self.mrdef_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split('|')
                if len(fields) >= 6:
                    cui, definition = fields[0], fields[5]

                    if cui in concepts and definition:
                        if not concepts[cui]['definition']:
                            concepts[cui]['definition'] = definition
                            definitions_added += 1

        return definitions_added

# ============================================================================
# TOP-N CONCEPT FILTER
# ============================================================================

def filter_to_top_concepts_per_diagnosis(found_concepts, term_to_cuis, required_terms, top_n=TOP_N_CONCEPTS_PER_DIAGNOSIS):
    """
    Filter to top-N most relevant concepts per diagnosis based on term match count
    """
    logger.info(f"🔍 Filtering to top-{top_n} concepts per diagnosis...")

    # Build diagnosis-specific concept scores
    diagnosis_concept_scores = {}

    for dx_code, terms_list in required_terms.items():
        concept_scores = Counter()

        # For each search term, give +1 score to matching concepts
        for term in terms_list:
            term_lower = term.lower().strip()
            if term_lower in term_to_cuis:
                for cui in term_to_cuis[term_lower]:
                    concept_scores[cui] += 1

        diagnosis_concept_scores[dx_code] = concept_scores

    # Select top-N per diagnosis
    filtered_whitelist = {}
    all_kept_cuis = set()

    for dx_code, concept_scores in diagnosis_concept_scores.items():
        # Get top-N by score
        top_concepts = [cui for cui, score in concept_scores.most_common(top_n)]
        filtered_whitelist[dx_code] = top_concepts
        all_kept_cuis.update(top_concepts)

        logger.info(f"  {dx_code}: {len(top_concepts)} concepts (was: {len(concept_scores)})")

    # Filter found_concepts to only include kept CUIs
    filtered_concepts = {
        cui: info for cui, info in found_concepts.items()
        if cui in all_kept_cuis
    }

    # Filter term_to_cuis
    filtered_term_to_cuis = {}
    for term, cuis in term_to_cuis.items():
        filtered_cuis = [cui for cui in cuis if cui in all_kept_cuis]
        if filtered_cuis:
            filtered_term_to_cuis[term] = filtered_cuis

    logger.info(f"  ✅ Filtered from {len(found_concepts)} to {len(filtered_concepts)} concepts")
    logger.info(f"  ✅ Expected labels per sample: ~{len(all_kept_cuis) / 4:.0f}-{len(all_kept_cuis) / 2:.0f}")

    return filtered_concepts, filtered_term_to_cuis

# ============================================================================
# MIMIC-IV DATA LOADER
# ============================================================================

class MIMICLoader:
    """MIMIC-IV data loader"""

    def __init__(self, mimic_path: Path, notes_path: Path):
        self.mimic_path = mimic_path
        self.hosp_path = mimic_path / 'mimic-iv-3.1/hosp'
        self.notes_path = notes_path

    def load_diagnoses(self) -> pd.DataFrame:
        diag_path = self.hosp_path / 'diagnoses_icd.csv.gz'
        return pd.read_csv(diag_path, compression='gzip')

    def load_admissions(self) -> pd.DataFrame:
        adm_path = self.hosp_path / 'admissions.csv.gz'
        return pd.read_csv(adm_path, compression='gzip')

    def load_discharge_notes(self) -> pd.DataFrame:
        discharge_path = self.notes_path / 'discharge.csv.gz'
        return pd.read_csv(discharge_path, compression='gzip')

def prepare_dataset(df_diag, df_notes, target_codes, max_per_code=MAX_SAMPLES_PER_DIAGNOSIS):
    """Prepare balanced dataset"""
    logger.info("🔧 Preparing dataset...")

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
        lambda codes: any(code in target_codes for code in codes)
    )
    df_filtered = df[df['has_target']].copy()

    df_filtered['labels'] = df_filtered['icd_codes'].apply(
        lambda codes: [1 if code in codes else 0 for code in target_codes]
    )

    # Balance dataset
    balanced_indices = set()
    for code in target_codes:
        code_indices = df_filtered[
            df_filtered['icd_codes'].apply(lambda x: code in x)
        ].index.tolist()
        n_samples = min(len(code_indices), max_per_code)
        selected = np.random.choice(code_indices, size=n_samples, replace=False)
        balanced_indices.update(selected)

    df_final = df_filtered.loc[list(balanced_indices)].reset_index(drop=True)
    df_final = df_final[df_final['text'].notnull()].reset_index(drop=True)

    logger.info(f"  ✅ Dataset: {len(df_final)} samples")
    return df_final

# ============================================================================
# CONCEPT STORE
# ============================================================================

class ConceptStore:
    """Build concept store from targeted UMLS concepts"""

    def __init__(self, umls_concepts: Dict, icd_to_cui: Dict):
        self.umls_concepts = umls_concepts
        self.icd_to_cui = icd_to_cui
        self.concepts = {}
        self.concept_to_idx = {}
        self.idx_to_concept = {}

    def build_from_targeted(self, target_codes: List[str], diagnosis_keywords: Dict):
        """Build concept store from already-loaded targeted concepts"""

        logger.info(f"🔬 Building concept store from {len(self.umls_concepts)} targeted concepts...")

        # Use ALL targeted concepts
        self.concepts = self.umls_concepts.copy()

        concept_list = list(self.concepts.keys())
        self.concept_to_idx = {cui: i for i, cui in enumerate(concept_list)}
        self.idx_to_concept = {i: cui for i, cui in enumerate(concept_list)}

        logger.info(f"  ✅ Stored {len(self.concepts)} concepts")

        # Build diagnosis-concept mappings
        self._build_diagnosis_mappings(target_codes, diagnosis_keywords)

        return self.concepts

    def _build_diagnosis_mappings(self, target_codes, diagnosis_keywords):
        """Map diagnoses to relevant concepts"""
        logger.info("🔗 Building diagnosis-concept mappings...")

        self.diagnosis_to_concepts = {}

        for dx_code in target_codes:
            keywords = diagnosis_keywords.get(dx_code, [])
            relevant_indices = []

            for cui, info in self.concepts.items():
                concept_idx = self.concept_to_idx[cui]
                terms_text = ' '.join(
                    [info['preferred_name']] + info.get('terms', [])
                ).lower()

                if any(kw in terms_text for kw in keywords):
                    relevant_indices.append(concept_idx)

            self.diagnosis_to_concepts[dx_code] = relevant_indices
            logger.info(f"  {dx_code}: {len(relevant_indices)} relevant concepts")

    def get_concepts_for_diagnosis(self, diagnosis_code: str) -> Dict:
        relevant_indices = self.diagnosis_to_concepts.get(diagnosis_code, [])
        return {
            self.idx_to_concept[idx]: self.concepts[self.idx_to_concept[idx]]
            for idx in relevant_indices
        }

    def create_concept_embeddings(self, tokenizer, model, device):
        logger.info("🧬 Creating concept embeddings...")

        concept_texts = []
        for cui, info in self.concepts.items():
            text = f"{info['preferred_name']}."
            if info.get('definition'):
                text += f" {info['definition'][:150]}"
            concept_texts.append(text)

        batch_size = 32
        all_embeddings = []

        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(concept_texts), batch_size), desc="  Encoding"):
                batch = concept_texts[i:i+batch_size]
                encodings = tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=128, return_tensors='pt'
                ).to(device)

                outputs = model(**encodings)
                embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(embeddings.cpu())

        final_embeddings = torch.cat(all_embeddings, dim=0).to(device)
        logger.info(f"  ✅ Created embeddings: {final_embeddings.shape}")

        return final_embeddings

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class EnhancedCrossAttention(nn.Module):
    """Cross-attention between clinical text and medical concepts"""

    def __init__(self, hidden_size, num_heads=8, dropout=DROPOUT):
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

    def __init__(self, base_model, num_concepts, num_classes, fusion_layers=FUSION_LAYERS):
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
        self.dropout = nn.Dropout(DROPOUT)

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

    def __init__(self, texts, labels, tokenizer, max_length=MAX_SEQUENCE_LENGTH, concept_labels=None):
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


class AlignmentLoss(nn.Module):
    """Alignment loss to enforce diagnosis-concept matching"""

    def __init__(self, concept_store, target_codes):
        super().__init__()
        self.concept_store = concept_store
        self.target_codes = target_codes
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, diagnosis_logits, concept_scores, diagnosis_labels, concept_labels):
        # Diagnosis loss
        diagnosis_loss = self.bce_loss(diagnosis_logits, diagnosis_labels)

        # Concept precision loss
        concept_precision_loss = self.bce_loss(concept_scores, concept_labels)

        # Confidence boost
        concept_probs = torch.sigmoid(concept_scores)
        top_k_probs = torch.topk(concept_probs, k=min(12, concept_probs.size(1)), dim=1)[0]
        confidence_loss = -torch.mean(top_k_probs)

        # Total loss
        total_loss = (
            LOSS_WEIGHT_DIAGNOSIS * diagnosis_loss +
            LOSS_WEIGHT_CONCEPT * concept_precision_loss +
            LOSS_WEIGHT_CONFIDENCE * confidence_loss
        )

        return total_loss, {
            'diagnosis': diagnosis_loss.item(),
            'concept': concept_precision_loss.item(),
            'confidence': confidence_loss.item()
        }

# ============================================================================
# WHITELIST LABELING
# ============================================================================

class TargetedWhitelistLabeler:
    """Generate labels using concepts found during targeted loading"""

    def __init__(self, concept_store, term_to_cuis, required_terms):
        self.concept_store = concept_store
        self.term_to_cuis = term_to_cuis
        self.required_terms = required_terms
        self.whitelist = {}

    def build_whitelist(self):
        """Build whitelist from targeted search results"""
        logger.info("📊 Building whitelist from search results...")

        for dx_code, terms_list in self.required_terms.items():
            whitelist_cuis = set()

            for term in terms_list:
                term_lower = term.lower().strip()
                if term_lower in self.term_to_cuis:
                    # Get CUIs that exist in concept store
                    for cui in self.term_to_cuis[term_lower]:
                        if cui in self.concept_store.concepts:
                            whitelist_cuis.add(cui)

            self.whitelist[dx_code] = list(whitelist_cuis)
            logger.info(f"  {dx_code}: {len(whitelist_cuis)} concepts")

        total = sum(len(v) for v in self.whitelist.values())
        logger.info(f"  ✅ Total whitelist concepts: {total}")

        return set([cui for cuis in self.whitelist.values() for cui in cuis])

    def generate_labels(self, diagnosis_codes: List[str]) -> List[int]:
        activated_cuis = set()

        for dx_code in diagnosis_codes:
            if dx_code in self.whitelist:
                activated_cuis.update(self.whitelist[dx_code])

        labels = []
        for cui in self.concept_store.concepts.keys():
            labels.append(1 if cui in activated_cuis else 0)

        return labels

    def generate_dataset_labels(self, df_data, cache_file=None):
        logger.info(f"🏷️  Generating labels for {len(df_data)} samples...")

        all_labels = []
        for row in tqdm(df_data.itertuples(), total=len(df_data), desc="  Labeling"):
            labels = self.generate_labels(row.icd_codes)
            all_labels.append(labels)

        all_labels = np.array(all_labels)

        if cache_file:
            with open(cache_file, 'wb') as f:
                pickle.dump(all_labels, f)

        avg_labels = all_labels.sum(axis=1).mean()
        logger.info(f"  ✅ Avg labels per sample: {avg_labels:.1f}")

        if 5 <= avg_labels <= 15:
            logger.info(f"  ✅ Labels in healthy range!")
        else:
            logger.warning(f"  ⚠️  Labels outside expected range (5-15)")

        return all_labels

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main(args):
    """Main training pipeline"""

    logger.info("="*80)
    logger.info("SHIFAMIND: MODEL TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Output Directory: {args.output_path}")
    logger.info(f"Checkpoint Directory: {CHECKPOINT_PATH}")

    # Validate data
    validate_data_structure()

    # Load UMLS concepts
    logger.info("="*70)
    logger.info("LOADING TARGETED CONCEPTS FROM UMLS")
    logger.info("="*70)

    targeted_loader = TargetedUMLSLoader(UMLS_PATH)
    umls_concepts_raw, term_to_cuis_raw, dx_coverage = targeted_loader.load_specific_concepts(REQUIRED_MEDICAL_TERMS)

    # Filter to top-N concepts
    umls_concepts, term_to_cuis = filter_to_top_concepts_per_diagnosis(
        umls_concepts_raw,
        term_to_cuis_raw,
        REQUIRED_MEDICAL_TERMS,
        top_n=TOP_N_CONCEPTS_PER_DIAGNOSIS
    )

    logger.info(f"✅ TARGETED LOADING COMPLETE:")
    logger.info(f"   Total concepts loaded: {len(umls_concepts)}")

    # Build ICD10 to CUI mapping
    icd10_to_cui = defaultdict(list)
    for cui, info in umls_concepts.items():
        if 'ICD10CM' in info['sources']:
            for code in info['sources']['ICD10CM']:
                icd10_to_cui[code].append(cui)

    logger.info(f"   ICD10 mappings: {len(icd10_to_cui)}")

    # Load MIMIC-IV data
    logger.info("="*70)
    logger.info("LOADING MIMIC-IV DATA")
    logger.info("="*70)

    mimic_loader = MIMICLoader(MIMIC_PATH, MIMIC_NOTES_PATH)
    df_diag = mimic_loader.load_diagnoses()
    df_notes = mimic_loader.load_discharge_notes()

    logger.info(f"✅ Loaded MIMIC-IV data:")
    logger.info(f"   Diagnoses: {len(df_diag)}")
    logger.info(f"   Notes: {len(df_notes)}")

    # Prepare dataset
    df_data = prepare_dataset(df_diag, df_notes, TARGET_CODES, max_per_code=args.max_samples_per_code)

    # Train/val/test split
    def get_primary_diagnosis(label_list):
        for i, val in enumerate(label_list):
            if val == 1:
                return i
        return 0

    df_data['primary_dx'] = df_data['labels'].apply(get_primary_diagnosis)

    df_train, df_temp = train_test_split(
        df_data, test_size=0.3, random_state=RANDOM_SEED, stratify=df_data['primary_dx']
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=df_temp['primary_dx']
    )

    df_train = df_train.drop('primary_dx', axis=1)
    df_val = df_val.drop('primary_dx', axis=1)
    df_test = df_test.drop('primary_dx', axis=1)

    logger.info(f"📊 Split:")
    logger.info(f"  Train: {len(df_train)}")
    logger.info(f"  Val: {len(df_val)}")
    logger.info(f"  Test: {len(df_test)}")

    # Initialize tokenizer and base model
    logger.info("Initializing Bio_ClinicalBERT...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = AutoModel.from_pretrained(BASE_MODEL_NAME).to(device)

    # Build concept store
    concept_store = ConceptStore(umls_concepts, icd10_to_cui)
    concept_store.build_from_targeted(TARGET_CODES, DIAGNOSIS_KEYWORDS)

    concept_embeddings = concept_store.create_concept_embeddings(tokenizer, base_model, device)

    logger.info("✅ Concept store complete")

    # Initialize model
    logger.info("="*70)
    logger.info("MODEL ARCHITECTURE")
    logger.info("="*70)

    shifamind_model = ShifaMindModel(
        base_model=base_model,
        num_concepts=len(concept_store.concepts),
        num_classes=len(TARGET_CODES),
        fusion_layers=FUSION_LAYERS
    ).to(device)

    logger.info(f"  Model parameters: {sum(p.numel() for p in shifamind_model.parameters()):,}")
    logger.info("✅ Model architecture defined and initialized")

    # Build whitelist
    logger.info("="*70)
    logger.info("WHITELIST LABELING FROM TARGETED CONCEPTS")
    logger.info("="*70)

    labeler = TargetedWhitelistLabeler(concept_store, term_to_cuis, REQUIRED_MEDICAL_TERMS)
    whitelist_concepts = labeler.build_whitelist()

    logger.info(f"✅ Whitelist ready: {len(whitelist_concepts)} unique concepts")

    # Generate concept labels
    logger.info("="*70)
    logger.info("GENERATING CONCEPT LABELS")
    logger.info("="*70)

    train_concept_labels = labeler.generate_dataset_labels(
        df_train,
        cache_file=str(args.output_path / 'concept_labels_train.pkl')
    )

    val_concept_labels = labeler.generate_dataset_labels(
        df_val,
        cache_file=str(args.output_path / 'concept_labels_val.pkl')
    )

    test_concept_labels = labeler.generate_dataset_labels(
        df_test,
        cache_file=str(args.output_path / 'concept_labels_test.pkl')
    )

    logger.info("✅ All concept labels generated")

    # ========================================================================
    # STAGE 1: DIAGNOSIS HEAD TRAINING
    # ========================================================================

    logger.info("="*70)
    logger.info("STAGE 1: DIAGNOSIS HEAD TRAINING")
    logger.info("="*70)

    if CHECKPOINT_STAGE1.exists() and not args.retrain:
        logger.info(f"✅ Found existing checkpoint: {CHECKPOINT_STAGE1}")
        logger.info("Skipping Stage 1 (already trained)")
        checkpoint = torch.load(CHECKPOINT_STAGE1, map_location=device)
        shifamind_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.info("Preparing data loaders...")
        train_dataset = ClinicalDataset(
            df_train['text'].tolist(),
            df_train['labels'].tolist(),
            tokenizer
        )
        val_dataset = ClinicalDataset(
            df_val['text'].tolist(),
            df_val['labels'].tolist(),
            tokenizer
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE)

        logger.info("Starting Stage 1 training...")
        optimizer = torch.optim.AdamW(shifamind_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss()

        num_training_steps = NUM_EPOCHS_STAGE1 * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_training_steps // 10,
            num_training_steps=num_training_steps
        )

        best_f1 = 0

        for epoch in range(NUM_EPOCHS_STAGE1):
            logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS_STAGE1}")

            shifamind_model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc="Training"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()

                outputs = shifamind_model(
                    input_ids, attention_mask, concept_embeddings,
                    return_diagnosis_only=True
                )

                loss = criterion(outputs['logits'], labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(shifamind_model.parameters(), max_norm=GRADIENT_CLIP_NORM)

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # Validation
            shifamind_model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = shifamind_model(
                        input_ids, attention_mask, concept_embeddings,
                        return_diagnosis_only=True
                    )

                    preds = torch.sigmoid(outputs['logits']).cpu().numpy()
                    all_preds.append(preds)
                    all_labels.append(labels.cpu().numpy())

            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)
            pred_binary = (all_preds > 0.5).astype(int)

            macro_f1 = f1_score(all_labels, pred_binary, average='macro', zero_division=0)

            logger.info(f"  Loss: {avg_loss:.4f}")
            logger.info(f"  Val Macro F1: {macro_f1:.4f}")

            if macro_f1 > best_f1:
                best_f1 = macro_f1

                torch.save({
                    'model_state_dict': shifamind_model.state_dict(),
                    'num_concepts': len(concept_store.concepts),
                    'concept_cuis': list(concept_store.concepts.keys()),
                    'concept_names': {cui: info['preferred_name'] for cui, info in concept_store.concepts.items()},
                    'concept_embeddings': concept_embeddings,
                    'macro_f1': best_f1
                }, CHECKPOINT_STAGE1)
                logger.info(f"  ✅ Saved checkpoint (F1: {best_f1:.4f})")

        logger.info(f"✅ Stage 1 complete. Best F1: {best_f1:.4f}")

    torch.cuda.empty_cache()

    # ========================================================================
    # STAGE 2: CONCEPT HEAD TRAINING
    # ========================================================================

    logger.info("="*70)
    logger.info("STAGE 2: CONCEPT HEAD TRAINING")
    logger.info("="*70)

    if CHECKPOINT_STAGE2.exists() and not args.retrain:
        logger.info(f"✅ Found existing checkpoint: {CHECKPOINT_STAGE2}")
        logger.info("Skipping Stage 2 (already trained)")
        checkpoint = torch.load(CHECKPOINT_STAGE2, map_location=device)
        shifamind_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.info("Preparing data loaders with concept labels...")
        train_dataset = ClinicalDataset(
            df_train['text'].tolist(),
            df_train['labels'].tolist(),
            tokenizer,
            concept_labels=train_concept_labels
        )
        val_dataset = ClinicalDataset(
            df_val['text'].tolist(),
            df_val['labels'].tolist(),
            tokenizer,
            concept_labels=val_concept_labels
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE)

        logger.info("Starting Stage 2 training...")
        optimizer = torch.optim.AdamW(shifamind_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss()

        best_concept_f1 = 0

        for epoch in range(NUM_EPOCHS_STAGE2):
            logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS_STAGE2}")

            shifamind_model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc="Training"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                concept_labels_batch = batch['concept_labels'].to(device)

                optimizer.zero_grad()

                outputs = shifamind_model(input_ids, attention_mask, concept_embeddings)

                # Concept loss
                concept_loss = criterion(outputs['concept_scores'], concept_labels_batch)

                # Confidence boost
                concept_probs = torch.sigmoid(outputs['concept_scores'])
                top_k_probs = torch.topk(concept_probs, k=min(12, concept_probs.size(1)), dim=1)[0]
                confidence_loss = -torch.mean(top_k_probs)

                loss = CONCEPT_LOSS_WEIGHT * concept_loss + CONFIDENCE_LOSS_WEIGHT * confidence_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(shifamind_model.parameters(), max_norm=GRADIENT_CLIP_NORM)

                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # Validation
            shifamind_model.eval()
            all_concept_preds = []
            all_concept_labels = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    concept_labels_batch = batch['concept_labels'].to(device)

                    outputs = shifamind_model(input_ids, attention_mask, concept_embeddings)

                    concept_preds = torch.sigmoid(outputs['concept_scores']).cpu().numpy()
                    all_concept_preds.append(concept_preds)
                    all_concept_labels.append(concept_labels_batch.cpu().numpy())

            all_concept_preds = np.vstack(all_concept_preds)
            all_concept_labels = np.vstack(all_concept_labels)
            concept_pred_binary = (all_concept_preds > 0.7).astype(int)

            concept_f1 = f1_score(all_concept_labels, concept_pred_binary, average='macro', zero_division=0)

            logger.info(f"  Loss: {avg_loss:.4f}")
            logger.info(f"  Val Concept F1: {concept_f1:.4f}")

            if concept_f1 > best_concept_f1:
                best_concept_f1 = concept_f1

                torch.save({
                    'model_state_dict': shifamind_model.state_dict(),
                    'num_concepts': len(concept_store.concepts),
                    'concept_cuis': list(concept_store.concepts.keys()),
                    'concept_names': {cui: info['preferred_name'] for cui, info in concept_store.concepts.items()},
                    'concept_embeddings': concept_embeddings,
                    'concept_f1': best_concept_f1
                }, CHECKPOINT_STAGE2)
                logger.info(f"  ✅ Saved checkpoint (F1: {best_concept_f1:.4f})")

        logger.info(f"✅ Stage 2 complete. Best Concept F1: {best_concept_f1:.4f}")

    torch.cuda.empty_cache()

    # ========================================================================
    # STAGE 3: JOINT FINE-TUNING
    # ========================================================================

    logger.info("="*70)
    logger.info("STAGE 3: JOINT FINE-TUNING WITH ALIGNMENT")
    logger.info("="*70)

    if CHECKPOINT_FINAL.exists() and not args.retrain:
        logger.info(f"✅ Found existing checkpoint: {CHECKPOINT_FINAL}")
        logger.info("Skipping Stage 3 (already trained)")
        checkpoint = torch.load(CHECKPOINT_FINAL, map_location=device)
        shifamind_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.info("Starting Stage 3 training...")

        optimizer = torch.optim.AdamW(shifamind_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = AlignmentLoss(concept_store, TARGET_CODES)

        num_training_steps = NUM_EPOCHS_STAGE3 * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_training_steps // 10,
            num_training_steps=num_training_steps
        )

        best_f1 = 0

        for epoch in range(NUM_EPOCHS_STAGE3):
            logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS_STAGE3}")

            shifamind_model.train()
            total_loss = 0
            loss_components = defaultdict(float)

            for batch in tqdm(train_loader, desc="Training"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                concept_labels_batch = batch['concept_labels'].to(device)

                optimizer.zero_grad()

                outputs = shifamind_model(input_ids, attention_mask, concept_embeddings)

                loss, components = criterion(
                    outputs['logits'],
                    outputs['concept_scores'],
                    labels,
                    concept_labels_batch
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(shifamind_model.parameters(), max_norm=GRADIENT_CLIP_NORM)

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                for k, v in components.items():
                    loss_components[k] += v

            avg_loss = total_loss / len(train_loader)

            logger.info(f"  Loss: {avg_loss:.4f}")
            for k, v in loss_components.items():
                logger.info(f"    {k}: {v/len(train_loader):.4f}")

            # Validation
            shifamind_model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = shifamind_model(input_ids, attention_mask, concept_embeddings)

                    preds = torch.sigmoid(outputs['logits']).cpu().numpy()
                    all_preds.append(preds)
                    all_labels.append(labels.cpu().numpy())

            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)
            pred_binary = (all_preds > 0.5).astype(int)

            macro_f1 = f1_score(all_labels, pred_binary, average='macro', zero_division=0)

            logger.info(f"  Val Macro F1: {macro_f1:.4f}")

            if macro_f1 > best_f1:
                best_f1 = macro_f1

                torch.save({
                    'model_state_dict': shifamind_model.state_dict(),
                    'num_concepts': len(concept_store.concepts),
                    'concept_cuis': list(concept_store.concepts.keys()),
                    'concept_names': {cui: info['preferred_name'] for cui, info in concept_store.concepts.items()},
                    'concept_embeddings': concept_embeddings,
                    'target_codes': TARGET_CODES,
                    'macro_f1': best_f1
                }, CHECKPOINT_FINAL)
                logger.info(f"  ✅ Saved checkpoint (F1: {best_f1:.4f})")

        logger.info(f"✅ Stage 3 complete. Best F1: {best_f1:.4f}")

    torch.cuda.empty_cache()

    # Save test data for evaluation
    test_cache = {
        'df_test': df_test,
        'test_concept_labels': test_concept_labels
    }
    with open(args.output_path / 'test_data_cache.pkl', 'wb') as f:
        pickle.dump(test_cache, f)

    logger.info("="*80)
    logger.info("✅ ALL TRAINING STAGES COMPLETE!")
    logger.info("="*80)
    logger.info(f"📊 Summary:")
    logger.info(f"   Concepts loaded: {len(umls_concepts)}")
    logger.info(f"   Whitelist concepts: {len(whitelist_concepts)}")
    logger.info(f"   Avg labels/sample: {train_concept_labels.sum(axis=1).mean():.1f}")
    logger.info(f"   Final F1: {best_f1:.4f}")
    logger.info(f"🎯 Training successful!")
    logger.info(f"   Next: Run final_evaluation.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ShifaMind model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--output-path',
        type=Path,
        default=RESULTS_PATH / 'experiments/training_run',
        help='Output directory for results'
    )

    parser.add_argument(
        '--max-samples-per-code',
        type=int,
        default=MAX_SAMPLES_PER_DIAGNOSIS,
        help='Maximum samples per diagnosis code'
    )

    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Retrain even if checkpoints exist'
    )

    # Use parse_known_args() to ignore Jupyter/Colab kernel arguments (e.g., -f)
    args, unknown = parser.parse_known_args()
    args.output_path.mkdir(parents=True, exist_ok=True)

    main(args)
