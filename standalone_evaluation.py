# ======================
# CONFIG
# ======================
API_KEY = "sk-proj-YOUR_KEY_HERE"   # <---- PUT YOUR OPENAI KEY HERE
N_SAMPLES = 100  # Number of samples to evaluate (increase for better metrics)
BASE_PATH = "/content/drive/MyDrive/ShifaMind"

# ======================
# IMPORTS
# ======================
import os, json, time, warnings, re, pickle
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix

# Setup
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
BASE_PATH = Path(BASE_PATH)
OUTPUT_PATH = BASE_PATH / '05_Comparisons'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TARGET_CODES = ['J189', 'I5023', 'A419', 'K8000']
ICD_DESCRIPTIONS = {
    'J189': 'Pneumonia, unspecified organism',
    'I5023': 'Acute on chronic systolic heart failure',
    'A419': 'Sepsis, unspecified organism',
    'K8000': 'Calculus of gallbladder with acute cholecystitis'
}
CODE_TO_LABEL = {code: i for i, code in enumerate(TARGET_CODES)}
LABEL_TO_CODE = {i: code for code, i in CODE_TO_LABEL.items()}

print("="*80)
print("SHIFAMIND COMPREHENSIVE EVALUATION WITH FULL METRICS")
print("="*80)
print(f"Device: {device}\nSamples: {N_SAMPLES}\nOutput: {OUTPUT_PATH}\n")

# ======================
# MODELS
# ======================
class EnhancedCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size, self.num_heads = hidden_size, num_heads
        self.head_dim = hidden_size // num_heads
        self.query, self.key, self.value = nn.Linear(hidden_size, hidden_size), nn.Linear(hidden_size, hidden_size), nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout, self.layer_norm = nn.Dropout(dropout), nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, concept_embeddings, attention_mask=None, return_attention=False):
        B, S, H = hidden_states.shape
        C = concept_embeddings.shape[0]
        concepts_batch = concept_embeddings.unsqueeze(0).expand(B, -1, -1)
        Q = self.query(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(concepts_batch).view(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(concepts_batch).view(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights_dropout = self.dropout(attention_weights)
        context = torch.matmul(attention_weights_dropout, V).transpose(1, 2).contiguous().view(B, S, H)
        gate_input = torch.cat([hidden_states, context], dim=-1)
        gate_values = torch.sigmoid(self.gate(gate_input))
        output = self.layer_norm(hidden_states + gate_values * context)
        return (output, attention_weights) if return_attention else (output, None)

class ShifaMindModel(nn.Module):
    def __init__(self, base_model, num_concepts, num_classes, fusion_layers=[9, 11]):
        super().__init__()
        self.base_model, self.fusion_layers = base_model, fusion_layers
        self.hidden_size = base_model.config.hidden_size
        self.fusion_modules = nn.ModuleList([EnhancedCrossAttention(self.hidden_size) for _ in fusion_layers])
        self.diagnosis_head = nn.Linear(self.hidden_size, num_classes)
        self.concept_head = nn.Linear(self.hidden_size, num_concepts)
        self.diagnosis_concept_interaction = nn.Bilinear(num_classes, num_concepts, num_concepts)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, concept_embeddings, return_attention=False):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        hidden_states, current_hidden = outputs.hidden_states, outputs.hidden_states[-1]
        fusion_attentions = {}
        for i, fusion_module in enumerate(self.fusion_modules):
            layer_hidden = hidden_states[self.fusion_layers[i]]
            fused_hidden, attn = fusion_module(layer_hidden, concept_embeddings, attention_mask, return_attention=return_attention)
            if return_attention and attn is not None:
                fusion_attentions[f'layer_{self.fusion_layers[i]}'] = attn
            if i == len(self.fusion_modules) - 1:
                current_hidden = fused_hidden
        cls_hidden = self.dropout(current_hidden[:, 0, :])
        diagnosis_logits, concept_logits = self.diagnosis_head(cls_hidden), self.concept_head(cls_hidden)
        diagnosis_probs = torch.sigmoid(diagnosis_logits)
        refined_concept_logits = self.diagnosis_concept_interaction(diagnosis_probs, torch.sigmoid(concept_logits))
        result = {'logits': diagnosis_logits, 'concept_scores': refined_concept_logits}
        if return_attention:
            fusion_attentions['input_ids'] = input_ids
            result['attention_weights'] = fusion_attentions
        return result

print("Loading models...")
checkpoint = torch.load(BASE_PATH / '03_Models/checkpoints/shifamind_model_final.pt', map_location=device)
concept_embeddings = checkpoint['concept_embeddings'].to(device)
num_concepts, concept_cuis, concept_names = checkpoint['num_concepts'], checkpoint['concept_cuis'], checkpoint['concept_names']
concept_store = {
    'concepts': {cui: {'preferred_name': name} for cui, name in concept_names.items()},
    'idx_to_concept': {i: cui for i, cui in enumerate(concept_cuis)}
}
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
shifamind_model = ShifaMindModel(base_model, num_concepts, len(TARGET_CODES), [9, 11]).to(device)
shifamind_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
shifamind_model.eval()

with open(BASE_PATH / '03_Models/clinical_knowledge_base_final.json', 'r') as f:
    knowledge_base = json.load(f)

bert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
bert_classifier = nn.Linear(768, len(TARGET_CODES)).to(device)
bert_model.eval(); bert_classifier.eval()

gpt_client = OpenAI(api_key=API_KEY)
print("✅ All models loaded\n")

# ======================
# HELPER FUNCTIONS
# ======================
def extract_evidence_spans(text, input_ids, attention_weights, concepts, tokenizer_obj, top_k=5, span_window=10):
    layer_9_attn, layer_11_attn = attention_weights.get('layer_9'), attention_weights.get('layer_11')
    if layer_9_attn is None or layer_11_attn is None:
        return []
    avg_attention = (layer_9_attn.squeeze(0).mean(0) + layer_11_attn.squeeze(0).mean(0)) / 2
    tokens = tokenizer_obj.convert_ids_to_tokens(input_ids.cpu().tolist())
    evidence_chains = []
    for concept in concepts[:top_k]:
        concept_idx = concept.get('idx', 0)
        if concept_idx >= avg_attention.shape[1]:
            continue
        concept_attention = avg_attention[:, concept_idx]
        topk_values, topk_indices = torch.topk(concept_attention, k=min(3, len(tokens)))
        spans = []
        for token_idx in topk_indices:
            token_idx = token_idx.item()
            start, end = max(0, token_idx - span_window), min(len(tokens), token_idx + span_window + 1)
            span_tokens = tokens[start:end]
            span_text = tokenizer_obj.convert_tokens_to_string(span_tokens)
            span_text = re.sub(r'(\[CLS\]|\[SEP\]|\[PAD\])', '', span_text).strip()
            span_text = re.sub(r'\s+', ' ', span_text).strip('.,;: ')
            if len(span_text) > 20:
                spans.append(span_text)
        unique_spans = list(dict.fromkeys(spans))[:2]
        evidence_chains.append({'concept': concept['name'], 'cui': concept.get('cui', 'UNKNOWN'),
                                'score': float(concept['score']), 'evidence_spans': unique_spans})
    return evidence_chains

def retrieve_clinical_knowledge(diagnosis_code, concepts, kb, top_k=2):
    if diagnosis_code not in kb:
        return []
    all_entries = kb[diagnosis_code]
    concept_keywords = set()
    for c in concepts[:10]:
        concept_keywords.update(c['name'].lower().split())
    scored_entries = []
    for entry in all_entries:
        entry_words = set(entry['text'].lower().split())
        overlap = len(concept_keywords & entry_words)
        keyword_bonus = sum(2 for kw in entry.get('keywords', []) if kw.lower() in concept_keywords)
        type_bonus = {'clinical_presentation': 3, 'diagnosis_description': 2}.get(entry['type'], 1)
        total_score = overlap + keyword_bonus + type_bonus
        scored_entries.append((total_score, entry))
    scored_entries.sort(reverse=True, key=lambda x: x[0])
    return [{'text': entry['text'], 'source': entry['source'], 'type': entry['type']}
            for score, entry in scored_entries[:top_k] if score > 0]

def calculate_calibration_error(y_true, y_pred_probs, n_bins=10):
    """Calculate Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_probs > bin_lower) & (y_pred_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred_probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

# ======================
# LOAD TEST DATA WITH GROUND TRUTH
# ======================
print("Loading test data with ground truth labels...")
test_cache_path = BASE_PATH / '04_Results/experiments/training_run/test_data_cache.pkl'

if test_cache_path.exists():
    with open(test_cache_path, 'rb') as f:
        test_cache = pickle.load(f)
    df_test = test_cache['df_test']
    test_concept_labels = test_cache['test_concept_labels']

    # Sample N_SAMPLES from test set
    if len(df_test) > N_SAMPLES:
        sample_indices = np.random.choice(len(df_test), N_SAMPLES, replace=False)
        df_test = df_test.iloc[sample_indices].reset_index(drop=True)
        test_concept_labels = test_concept_labels[sample_indices]

    test_notes = df_test['text'].tolist()
    y_true = []
    for _, row in df_test.iterrows():
        label = [0] * len(TARGET_CODES)
        for code in TARGET_CODES:
            if code in row['icd_codes']:
                label[CODE_TO_LABEL[code]] = 1
        y_true.append(label)
    y_true = np.array(y_true)

    print(f"✅ Loaded {len(test_notes)} test notes with ground truth labels")
    print(f"   Label distribution: {y_true.sum(axis=0)}")
else:
    print("⚠️  Warning: test_data_cache.pkl not found. Metrics will not be computed.")
    print("   Run final_model_training.py first to generate test cache.")
    exit(1)

print()

# ======================
# RUN PREDICTIONS WITH FULL EXPLAINABILITY
# ======================
print("Running predictions...")
print("="*80 + "\n")

results = {'shifamind': [], 'bioclinbert': [], 'gpt4': []}
predictions_probs = {'shifamind': [], 'bioclinbert': [], 'gpt4': []}
times = {'shifamind': [], 'bioclinbert': [], 'gpt4': []}
costs, example_predictions = [], []

# Track explainability metrics for ShifaMind
shifamind_concepts_per_diagnosis = []
shifamind_has_evidence = []

for i, (text, true_label) in enumerate(tqdm(list(zip(test_notes, y_true)), desc="Evaluating"), 1):
    # === SHIFAMIND - FULL EXPLAINABILITY ===
    start = time.time()
    encoded = tokenizer(text[:2000], return_tensors='pt', padding='max_length', truncation=True, max_length=384).to(device)
    with torch.no_grad():
        output = shifamind_model(encoded['input_ids'], encoded['attention_mask'], concept_embeddings, return_attention=True)

    probs_sm = torch.sigmoid(output['logits']).cpu().numpy()[0]
    pred_label_sm = int(np.argmax(probs_sm))
    conf_sm = float(probs_sm[pred_label_sm])
    pred_code_sm = LABEL_TO_CODE[pred_label_sm]
    time_sm = time.time() - start

    # Extract concepts
    concept_scores = torch.sigmoid(output['concept_scores']).cpu().numpy()[0]
    top_concept_indices = np.argsort(concept_scores)[::-1][:10]
    top_concepts = []
    for idx in top_concept_indices:
        cui = concept_store['idx_to_concept'][idx]
        name = concept_store['concepts'][cui]['preferred_name']
        score = float(concept_scores[idx])
        top_concepts.append({'idx': int(idx), 'cui': cui, 'name': name, 'score': score})

    # Evidence spans
    evidence_chains = extract_evidence_spans(text[:2000], encoded['input_ids'][0], output['attention_weights'], top_concepts, tokenizer, top_k=5)

    # Clinical knowledge
    clinical_knowledge = retrieve_clinical_knowledge(pred_code_sm, top_concepts, knowledge_base, top_k=3)

    results['shifamind'].append(probs_sm)
    predictions_probs['shifamind'].append(probs_sm)
    times['shifamind'].append(time_sm)

    # Track explainability metrics
    shifamind_concepts_per_diagnosis.append(len(top_concepts))
    shifamind_has_evidence.append(1 if evidence_chains else 0)

    # === BIO_CLINICALBERT BASELINE ===
    start = time.time()
    encoded_bert = bert_tokenizer(text[:2000], return_tensors='pt', padding='max_length', truncation=True, max_length=384).to(device)
    with torch.no_grad():
        bert_output = bert_model(**encoded_bert)
        bert_logits = bert_classifier(bert_output.pooler_output)
    probs_bert = torch.sigmoid(bert_logits).cpu().numpy()[0]
    pred_label_bert = int(np.argmax(probs_bert))
    conf_bert = float(probs_bert[pred_label_bert])
    pred_code_bert = LABEL_TO_CODE[pred_label_bert]
    time_bert = time.time() - start

    results['bioclinbert'].append(probs_bert)
    predictions_probs['bioclinbert'].append(probs_bert)
    times['bioclinbert'].append(time_bert)

    # === GPT-4O-MINI ===
    try:
        start = time.time()
        prompt = f"""You are a medical diagnosis assistant. Given this discharge summary, predict ONE primary diagnosis from these options:
- J189: Pneumonia
- I5023: Heart Failure
- A419: Sepsis
- K8000: Cholecystitis

Discharge Summary:
{text[:1500]}

Respond with ONLY the ICD-10 code and confidence (0-100%). Format: CODE CONFIDENCE%
Example: J189 85%"""

        response = gpt_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
                                                        max_tokens=50, temperature=0.3)
        result_text = response.choices[0].message.content.strip()
        time_gpt = time.time() - start

        # Parse response
        match = re.search(r'([IJKA]\d+)\s+(\d+)%', result_text)
        if match:
            pred_code_gpt = match.group(1)
            conf_gpt = float(match.group(2)) / 100.0
            pred_label_gpt = CODE_TO_LABEL.get(pred_code_gpt, -1)
            if pred_label_gpt >= 0:
                probs_gpt = np.zeros(len(TARGET_CODES))
                probs_gpt[pred_label_gpt] = conf_gpt
                results['gpt4'].append(probs_gpt)
                predictions_probs['gpt4'].append(probs_gpt)
            else:
                results['gpt4'].append(np.zeros(len(TARGET_CODES)))
                predictions_probs['gpt4'].append(np.zeros(len(TARGET_CODES)))
        else:
            results['gpt4'].append(np.zeros(len(TARGET_CODES)))
            predictions_probs['gpt4'].append(np.zeros(len(TARGET_CODES)))

        times['gpt4'].append(time_gpt)
        costs.append(0.00015 * len(text.split()) / 1000 + 0.0006 * 50 / 1000)
    except Exception as e:
        results['gpt4'].append(np.zeros(len(TARGET_CODES)))
        predictions_probs['gpt4'].append(np.zeros(len(TARGET_CODES)))
        times['gpt4'].append(0)

    # Save first 2 examples for detailed output
    if i <= 2:
        example_predictions.append({
            'example_num': i,
            'clinical_note': text[:500] + "...",
            'ground_truth': [LABEL_TO_CODE[idx] for idx, val in enumerate(true_label) if val == 1],
            'shifamind': {
                'prediction': f"{pred_code_sm} - {ICD_DESCRIPTIONS[pred_code_sm]}",
                'confidence': f"{conf_sm:.1%}",
                'time': f"{time_sm:.3f}s",
                'top_concepts': top_concepts[:5],
                'evidence_chains': evidence_chains[:3],
                'clinical_knowledge': clinical_knowledge
            },
            'bioclinbert': {
                'prediction': f"{pred_code_bert} - {ICD_DESCRIPTIONS[pred_code_bert]}",
                'confidence': f"{conf_bert:.1%}",
                'time': f"{time_bert:.3f}s",
                'explainability': "❌ No explainability (black box)"
            },
            'gpt4': {
                'prediction': f"{pred_code_gpt} - {ICD_DESCRIPTIONS.get(pred_code_gpt, 'Unknown')}" if 'pred_code_gpt' in locals() else "N/A",
                'confidence': f"{conf_gpt:.1%}" if 'conf_gpt' in locals() else "N/A",
                'time': f"{time_gpt:.3f}s" if 'time_gpt' in locals() else "N/A",
                'explainability': "⚠️  Basic text generation only"
            }
        })

print("\n✅ Predictions complete\n")

# ======================
# COMPUTE COMPREHENSIVE METRICS
# ======================
print("="*80)
print("COMPUTING COMPREHENSIVE METRICS")
print("="*80)

# Convert to numpy arrays
y_pred_shifamind = np.array(predictions_probs['shifamind'])
y_pred_bioclinbert = np.array(predictions_probs['bioclinbert'])
y_pred_gpt4 = np.array(predictions_probs['gpt4'])

# Predictions (binary)
y_pred_binary_sm = (y_pred_shifamind > 0.5).astype(int)
y_pred_binary_bc = (y_pred_bioclinbert > 0.5).astype(int)
y_pred_binary_gpt = (y_pred_gpt4 > 0.5).astype(int)

# === CORE METRICS ===
metrics_summary = {}

for model_name, y_pred_binary, y_pred_probs in [
    ('ShifaMind', y_pred_binary_sm, y_pred_shifamind),
    ('Bio_ClinicalBERT', y_pred_binary_bc, y_pred_bioclinbert),
    ('GPT-4o-mini', y_pred_binary_gpt, y_pred_gpt4)
]:
    # Macro F1
    macro_f1 = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)

    # Micro F1
    micro_f1 = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)

    # AUROC (only if we have positive samples for each class)
    try:
        auroc = roc_auc_score(y_true, y_pred_probs, average='macro')
    except:
        auroc = 0.0

    # Per-diagnosis F1
    per_diag_f1 = {}
    for idx, code in enumerate(TARGET_CODES):
        f1 = f1_score(y_true[:, idx], y_pred_binary[:, idx], zero_division=0)
        per_diag_f1[code] = f1

    metrics_summary[model_name] = {
        'Macro F1': macro_f1,
        'Micro F1': micro_f1,
        'AUROC': auroc,
        'Per-diagnosis F1': per_diag_f1
    }

# === EXPLAINABILITY METRICS (ShifaMind only) ===
# Citation Completeness: % of predictions with evidence
citation_completeness = np.mean(shifamind_has_evidence)

# Avg Concepts per Diagnosis
avg_concepts_per_diagnosis = np.mean(shifamind_concepts_per_diagnosis)

# Calibration Error
y_pred_conf = y_pred_shifamind.max(axis=1)
y_true_conf = y_true[np.arange(len(y_true)), y_pred_shifamind.argmax(axis=1)]
calibration_error = calculate_calibration_error(y_true_conf, y_pred_conf, n_bins=10)

explainability_metrics = {
    'Citation Completeness': citation_completeness,
    'Avg Concepts/Diagnosis': avg_concepts_per_diagnosis,
    'Calibration Error (ECE)': calibration_error
}

# Print metrics
print("\n" + "="*80)
print("📊 CORE METRICS COMPARISON")
print("="*80)
print(f"\n{'Model':<25} {'Macro F1':<12} {'Micro F1':<12} {'AUROC':<12}")
print("-" * 61)
for model_name in ['ShifaMind', 'Bio_ClinicalBERT', 'GPT-4o-mini']:
    m = metrics_summary[model_name]
    print(f"{model_name:<25} {m['Macro F1']:<12.4f} {m['Micro F1']:<12.4f} {m['AUROC']:<12.4f}")

print("\n" + "="*80)
print("🔬 SHIFAMIND EXPLAINABILITY METRICS")
print("="*80)
print(f"Citation Completeness:     {explainability_metrics['Citation Completeness']:.2%}")
print(f"Avg Concepts/Diagnosis:    {explainability_metrics['Avg Concepts/Diagnosis']:.1f}")
print(f"Calibration Error (ECE):   {explainability_metrics['Calibration Error (ECE)']:.4f}")

print("\n" + "="*80)
print("📈 PER-DIAGNOSIS F1 SCORES (ShifaMind)")
print("="*80)
for code in TARGET_CODES:
    f1 = metrics_summary['ShifaMind']['Per-diagnosis F1'][code]
    print(f"{code} ({ICD_DESCRIPTIONS[code]:<50}): {f1:.4f}")

print()

# ======================
# SAVE METRICS
# ======================
print("Saving comprehensive metrics...")

# Save to CSV
metrics_df = pd.DataFrame({
    'Model': ['ShifaMind', 'Bio_ClinicalBERT', 'GPT-4o-mini'],
    'Macro F1': [metrics_summary[m]['Macro F1'] for m in ['ShifaMind', 'Bio_ClinicalBERT', 'GPT-4o-mini']],
    'Micro F1': [metrics_summary[m]['Micro F1'] for m in ['ShifaMind', 'Bio_ClinicalBERT', 'GPT-4o-mini']],
    'AUROC': [metrics_summary[m]['AUROC'] for m in ['ShifaMind', 'Bio_ClinicalBERT', 'GPT-4o-mini']],
    'Avg Inference Time (s)': [np.mean(times['shifamind']), np.mean(times['bioclinbert']),
                                 np.mean(times['gpt4']) if times['gpt4'] else 0]
})
metrics_df.to_csv(OUTPUT_PATH / 'comprehensive_metrics.csv', index=False)
print("✅ comprehensive_metrics.csv")

# Save explainability metrics
explainability_df = pd.DataFrame({
    'Metric': ['Citation Completeness', 'Avg Concepts/Diagnosis', 'Calibration Error (ECE)'],
    'Value': [explainability_metrics['Citation Completeness'],
              explainability_metrics['Avg Concepts/Diagnosis'],
              explainability_metrics['Calibration Error (ECE)']]
})
explainability_df.to_csv(OUTPUT_PATH / 'shifamind_explainability_metrics.csv', index=False)
print("✅ shifamind_explainability_metrics.csv")

# Save per-diagnosis F1
per_diag_df = pd.DataFrame({
    'Diagnosis Code': TARGET_CODES,
    'Description': [ICD_DESCRIPTIONS[c] for c in TARGET_CODES],
    'F1 Score': [metrics_summary['ShifaMind']['Per-diagnosis F1'][c] for c in TARGET_CODES]
})
per_diag_df.to_csv(OUTPUT_PATH / 'shifamind_per_diagnosis_f1.csv', index=False)
print("✅ shifamind_per_diagnosis_f1.csv")

# ======================
# VISUALIZATIONS
# ======================
print("\nGenerating visualizations...")

# 1. Metrics Comparison Bar Chart
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
model_names = ['ShifaMind', 'Bio_ClinicalBERT', 'GPT-4o-mini']
colors = ['#2ecc71', '#3498db', '#e74c3c']

for idx, (metric, ax) in enumerate(zip(['Macro F1', 'Micro F1', 'AUROC'], axes)):
    values = [metrics_summary[m][metric] for m in model_names]
    bars = ax.bar(model_names, values, color=colors, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ metrics_comparison.png")

# 2. Per-Diagnosis F1 Scores
fig, ax = plt.subplots(figsize=(12, 6))
diagnosis_labels = [f"{code}\n{ICD_DESCRIPTIONS[code][:20]}" for code in TARGET_CODES]
f1_scores = [metrics_summary['ShifaMind']['Per-diagnosis F1'][c] for c in TARGET_CODES]
bars = ax.bar(diagnosis_labels, f1_scores, color='#2ecc71', alpha=0.8, edgecolor='black')
for bar, score in zip(bars, f1_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('ShifaMind: Per-Diagnosis F1 Scores', fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'per_diagnosis_f1.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ per_diagnosis_f1.png")

# 3. Explainability Metrics
fig, ax = plt.subplots(figsize=(10, 6))
expl_metrics = ['Citation\nCompleteness', 'Avg Concepts\nper Diagnosis', 'Calibration\nError (ECE)']
expl_values = [explainability_metrics['Citation Completeness'],
               explainability_metrics['Avg Concepts/Diagnosis'] / 20,  # Normalize to 0-1
               explainability_metrics['Calibration Error (ECE)']]
expl_colors = ['#27ae60', '#3498db', '#e67e22']
bars = ax.bar(expl_metrics, expl_values, color=expl_colors, alpha=0.8, edgecolor='black')

# Add actual values on bars
labels_actual = [f"{explainability_metrics['Citation Completeness']:.1%}",
                f"{explainability_metrics['Avg Concepts/Diagnosis']:.1f}",
                f"{explainability_metrics['Calibration Error (ECE)']:.4f}"]
for bar, label in zip(bars, labels_actual):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            label, ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('Metric Value (normalized)', fontsize=12, fontweight='bold')
ax.set_title('ShifaMind: Explainability Metrics', fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, 1.2)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'explainability_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ explainability_metrics.png")

# 4. Inference Time Comparison
fig, ax = plt.subplots(figsize=(10, 6))
avg_times = [np.mean(times['shifamind']), np.mean(times['bioclinbert']), np.mean(times['gpt4']) if times['gpt4'] else 0]
bars = ax.bar(model_names, avg_times, color=colors, alpha=0.8, edgecolor='black')
for bar, time_val in zip(bars, avg_times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{time_val:.3f}s',
            ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Average Inference Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, max(avg_times) * 1.2)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'inference_time_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ inference_time_comparison.png")

# ======================
# SAVE EXAMPLE PREDICTIONS WITH FULL EXPLAINABILITY
# ======================
print("\nSaving example predictions with full explainability...")
with open(OUTPUT_PATH / 'example_predictions_detailed.md', 'w') as f:
    f.write("# ShifaMind Model Comparison - Example Predictions with Full Explainability\n\n")
    f.write("This demonstrates ShifaMind's unique explainability features compared to baseline models.\n\n")
    f.write("---\n\n")

    for ex in example_predictions:
        f.write(f"## Example {ex['example_num']}\n\n")
        f.write("### Clinical Note:\n\n")
        f.write(f"```\n{ex['clinical_note']}\n```\n\n")
        f.write(f"**Ground Truth:** {', '.join(ex['ground_truth'])}\n\n")
        f.write("---\n\n")

        f.write("### 🤖 ShifaMind Prediction (WITH FULL EXPLAINABILITY)\n\n")
        sm = ex['shifamind']
        f.write(f"**Diagnosis:** {sm['prediction']}\n\n")
        f.write(f"**Confidence:** {sm['confidence']}\n\n")
        f.write(f"**Inference Time:** {sm['time']}\n\n")

        f.write("#### 🔬 Top Medical Concepts Extracted:\n\n")
        for i, concept in enumerate(sm['top_concepts'], 1):
            f.write(f"{i}. **{concept['name']}** ({concept['score']:.2%})\n")
        f.write("\n")

        if sm['evidence_chains']:
            f.write("#### 📋 Evidence Chains (Supporting Quotes from Text):\n\n")
            for chain in sm['evidence_chains']:
                f.write(f"**Concept: {chain['concept']}** (Score: {chain['score']:.1%})\n\n")
                if chain['evidence_spans']:
                    for span in chain['evidence_spans']:
                        f.write(f"> \"{span}\"\n\n")
                else:
                    f.write("> *(No direct text evidence)*\n\n")

        if sm['clinical_knowledge']:
            f.write("#### 📚 Clinical Knowledge Retrieved:\n\n")
            for i, kb_entry in enumerate(sm['clinical_knowledge'], 1):
                f.write(f"{i}. **{kb_entry['type'].replace('_', ' ').title()}**\n\n")
                f.write(f"   {kb_entry['text']}\n\n")
                f.write(f"   *Source: {kb_entry['source']}*\n\n")

        f.write("---\n\n")

        f.write("### 🔵 Bio_ClinicalBERT Prediction (BASELINE)\n\n")
        bc = ex['bioclinbert']
        f.write(f"**Diagnosis:** {bc['prediction']}\n\n")
        f.write(f"**Confidence:** {bc['confidence']}\n\n")
        f.write(f"**Inference Time:** {bc['time']}\n\n")
        f.write(f"**Explainability:** {bc['explainability']}\n\n")
        f.write("---\n\n")

        f.write("### 🔴 GPT-4o-mini Prediction (API)\n\n")
        gpt = ex['gpt4']
        f.write(f"**Diagnosis:** {gpt['prediction']}\n\n")
        f.write(f"**Confidence:** {gpt['confidence']}\n\n")
        f.write(f"**Inference Time:** {gpt['time']}\n\n")
        f.write(f"**Explainability:** {gpt['explainability']}\n\n")
        f.write("\n---\n\n")

print("✅ example_predictions_detailed.md")

# ======================
# FINAL SUMMARY
# ======================
print("\n" + "="*80)
print("✅ COMPREHENSIVE EVALUATION COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: {OUTPUT_PATH}")
print("\n📊 Generated Files:")
print("  • comprehensive_metrics.csv - Core metrics for all models")
print("  • shifamind_explainability_metrics.csv - Explainability metrics")
print("  • shifamind_per_diagnosis_f1.csv - Per-diagnosis F1 scores")
print("  • metrics_comparison.png - Macro/Micro F1 and AUROC comparison")
print("  • per_diagnosis_f1.png - Per-diagnosis F1 bar chart")
print("  • explainability_metrics.png - Citation completeness, concepts, calibration")
print("  • inference_time_comparison.png - Speed comparison")
print("  • example_predictions_detailed.md - 2 examples with full explainability")
print("\n" + "="*80)
