# ======================
# CONFIG
# ======================
API_KEY = "sk-proj-YOUR_KEY_HERE"   # <---- PUT YOUR OPENAI KEY HERE
N_SAMPLES = 30  # Number of samples to test
BASE_PATH = "/content/drive/MyDrive/ShifaMind"

# ======================
# SCRIPT
# ======================
import os
import json
import time
import warnings
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

# Setup
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
BASE_PATH = Path(BASE_PATH)
OUTPUT_PATH = BASE_PATH / '05_Comparisons'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TARGET_CODES = ['J189', 'I5023', 'A419', 'K8000']
ICD_DESCRIPTIONS = {'J189': 'Pneumonia', 'I5023': 'Heart Failure', 'A419': 'Sepsis', 'K8000': 'Cholecystitis'}

print("="*80)
print("SHIFAMIND COMPREHENSIVE EVALUATION")
print("="*80)
print(f"Device: {device}")
print(f"Samples: {N_SAMPLES}")
print(f"Output: {OUTPUT_PATH}\n")

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class EnhancedCrossAttention(nn.Module):
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
        batch_size, seq_len, _ = hidden_states.shape
        num_concepts = concept_embeddings.shape[0]
        concepts_batch = concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        gate_input = torch.cat([hidden_states, context], dim=-1)
        gate_values = torch.sigmoid(self.gate(gate_input))
        return self.layer_norm(hidden_states + gate_values * context), attn_weights.mean(dim=1)

class ShifaMindModel(nn.Module):
    def __init__(self, base_model, num_concepts, num_classes, fusion_layers=[9, 11]):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.fusion_layers = fusion_layers
        self.fusion_modules = nn.ModuleList([EnhancedCrossAttention(self.hidden_size) for _ in fusion_layers])
        self.diagnosis_head = nn.Linear(self.hidden_size, num_classes)
        self.concept_head = nn.Linear(self.hidden_size, num_concepts)
        self.diagnosis_concept_interaction = nn.Bilinear(num_classes, num_concepts, num_concepts)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, concept_embeddings):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        current_hidden = hidden_states[-1]
        for i, fusion_module in enumerate(self.fusion_modules):
            layer_hidden = hidden_states[self.fusion_layers[i]]
            fused_hidden, _ = fusion_module(layer_hidden, concept_embeddings, attention_mask)
            if i == len(self.fusion_modules) - 1:
                current_hidden = fused_hidden
        cls_hidden = self.dropout(current_hidden[:, 0, :])
        diagnosis_logits = self.diagnosis_head(cls_hidden)
        concept_logits = self.concept_head(cls_hidden)
        diagnosis_probs = torch.sigmoid(diagnosis_logits)
        refined_concept_logits = self.diagnosis_concept_interaction(diagnosis_probs, torch.sigmoid(concept_logits))
        return {'logits': diagnosis_logits, 'concept_scores': refined_concept_logits}

# ============================================================================
# LOAD MODELS
# ============================================================================

print("Loading ShifaMind...")
checkpoint = torch.load(BASE_PATH / '03_Models/checkpoints/shifamind_model.pt', map_location=device)
concept_embeddings = checkpoint['concept_embeddings'].to(device)
num_concepts = checkpoint['num_concepts']
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
shifamind_model = ShifaMindModel(base_model, num_concepts, len(TARGET_CODES), [9, 11]).to(device)
shifamind_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
shifamind_model.eval()
print("✅ ShifaMind loaded\n")

print("Loading Bio_ClinicalBERT baseline...")
bert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
bert_classifier = nn.Linear(768, len(TARGET_CODES)).to(device)
bert_model.eval()
bert_classifier.eval()
print("✅ Bio_ClinicalBERT loaded\n")

print("Setting up GPT-4o-mini...")
gpt_client = OpenAI(api_key=API_KEY)
print("✅ GPT-4o-mini ready\n")

# ============================================================================
# LOAD TEST DATA
# ============================================================================

print("Loading test data...")
notes_path = BASE_PATH / "01_Raw_Datasets/Extracted/mimic-iv-note-2.2/note/discharge.csv.gz"
df = pd.read_csv(notes_path, compression="gzip", nrows=500, on_bad_lines="skip")
df = df[df["text"].str.len() > 500].sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)
test_notes = df['text'].tolist()
print(f"✅ Loaded {len(test_notes)} test notes\n")

# ============================================================================
# RUN PREDICTIONS
# ============================================================================

print("Running predictions...")
print("="*80 + "\n")

results = {'shifamind': [], 'bioclinbert': [], 'gpt4': []}
times = {'shifamind': [], 'bioclinbert': [], 'gpt4': []}
costs = []

for i, text in enumerate(tqdm(test_notes), 1):
    # ShifaMind
    start = time.time()
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=384, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = shifamind_model(encoding['input_ids'], encoding['attention_mask'], concept_embeddings)
        probs_sm = torch.sigmoid(outputs['logits']).cpu().numpy()[0]
    times['shifamind'].append(time.time() - start)
    results['shifamind'].append(probs_sm)

    # Bio_ClinicalBERT
    start = time.time()
    encoding = bert_tokenizer(text, padding='max_length', truncation=True, max_length=384, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = bert_model(**encoding)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        logits = bert_classifier(cls_hidden)
        probs_bc = torch.sigmoid(logits).cpu().numpy()[0]
    times['bioclinbert'].append(time.time() - start)
    results['bioclinbert'].append(probs_bc)

    # GPT-4 (limit to 20 calls to save cost)
    if i <= 20:
        prompt = f"""You are a clinical diagnosis AI. Analyze this note and predict the primary diagnosis.

Clinical Note:
{text[:2000]}

Respond ONLY with JSON:
{{
  "diagnosis": "Pneumonia" or "Heart Failure" or "Sepsis" or "Cholecystitis",
  "confidence": 0-100
}}"""

        try:
            start = time.time()
            response = gpt_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )
            latency = time.time() - start

            content = response.choices[0].message.content
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()

            try:
                result = json.loads(content)
            except:
                result = {"diagnosis": "Pneumonia", "confidence": 50}

            diagnosis = result.get("diagnosis", "Pneumonia")
            confidence = result.get("confidence", 50) / 100.0

            probs_gpt = np.zeros(len(TARGET_CODES))
            for j, desc in enumerate(ICD_DESCRIPTIONS.values()):
                if desc == diagnosis:
                    probs_gpt[j] = confidence
                else:
                    probs_gpt[j] = (1 - confidence) / (len(TARGET_CODES) - 1)

            usage = response.usage
            cost = (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60) / 1_000_000

            times['gpt4'].append(latency)
            results['gpt4'].append(probs_gpt)
            costs.append(cost)
            time.sleep(0.5)
        except Exception as e:
            print(f"GPT Error: {e}")

print("\n✅ Predictions complete\n")

# ============================================================================
# GENERATE VISUALIZATIONS
# ============================================================================

print("Generating visualizations...")

# 1. Inference Time
fig, ax = plt.subplots(figsize=(10, 6))
models = ['ShifaMind', 'Bio_ClinicalBERT', 'GPT-4o-mini']
avg_times = [np.mean(times['shifamind']), np.mean(times['bioclinbert']), np.mean(times['gpt4']) if times['gpt4'] else 0]
colors = ['#2ecc71', '#3498db', '#e74c3c']
bars = ax.bar(models, avg_times, color=colors, alpha=0.8, edgecolor='black')
for bar, time_val in zip(bars, avg_times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{time_val:.3f}s', ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Average Inference Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, max(avg_times) * 1.2)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'inference_time_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ inference_time_comparison.png")

# 2. Cost
fig, ax = plt.subplots(figsize=(10, 6))
total_cost_gpt = sum(costs) if costs else 0
cost_per_1k = (total_cost_gpt / len(costs) * 1000) if costs else 0
models_cost = ['ShifaMind\n(Local)', 'Bio_ClinicalBERT\n(Local)', f'GPT-4o-mini\n(${cost_per_1k:.2f}/1k calls)']
cost_values = [0, 0, cost_per_1k]
bars = ax.bar(models_cost, cost_values, color=colors, alpha=0.8, edgecolor='black')
for i, bar in enumerate(bars):
    if i < 2:
        ax.text(bar.get_x() + bar.get_width()/2, 0.01, 'FREE', ha='center', va='bottom', fontweight='bold', fontsize=12)
    else:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'${cost_values[i]:.2f}', ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Cost per 1000 Predictions ($)', fontsize=12, fontweight='bold')
ax.set_title('Cost Comparison', fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, max(cost_values) * 1.3 if max(cost_values) > 0 else 1)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'cost_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ cost_comparison.png")

# 3. Confidence Distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, (model, ax) in enumerate(zip(['shifamind', 'bioclinbert', 'gpt4'], axes)):
    if results[model]:
        confidences = [np.max(pred) for pred in results[model]]
        ax.hist(confidences, bins=20, color=colors[idx], alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(confidences):.2f}')
        ax.set_xlabel('Prediction Confidence', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(['ShifaMind', 'Bio_ClinicalBERT', 'GPT-4o-mini'][idx], fontweight='bold')
        ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'confidence_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ confidence_distributions.png")

# 4. Capabilities Matrix
fig, ax = plt.subplots(figsize=(12, 8))
capabilities = ['Diagnosis Prediction', 'Confidence Scores', 'Medical Concept Extraction', 'Evidence Spans',
                'Clinical Knowledge Retrieval', 'Offline Operation', 'HIPAA Compliant', 'Explainability']
matrix = np.array([[1,1,1], [1,1,1], [1,0,0], [1,0,0], [1,0,0], [1,1,0], [1,1,0], [1,0,0]])
im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(np.arange(3))
ax.set_yticks(np.arange(len(capabilities)))
ax.set_xticklabels(['ShifaMind', 'Bio_ClinicalBERT', 'GPT-4o-mini'], fontweight='bold')
ax.set_yticklabels(capabilities)
for i in range(len(capabilities)):
    for j in range(3):
        ax.text(j, i, '✓' if matrix[i, j] == 1 else '✗', ha="center", va="center", color="black", fontsize=16, fontweight='bold')
ax.set_title('Model Capabilities Comparison', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'capabilities_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ capabilities_matrix.png")

# ============================================================================
# GENERATE TABLE
# ============================================================================

print("\nGenerating comparison table...")
summary_data = {
    'Model': ['ShifaMind', 'Bio_ClinicalBERT', 'GPT-4o-mini'],
    'Avg Inference Time (s)': [f"{np.mean(times['shifamind']):.3f}", f"{np.mean(times['bioclinbert']):.3f}",
                                 f"{np.mean(times['gpt4']):.3f}" if times['gpt4'] else 'N/A'],
    'Cost per 1k Calls': ['$0.00', '$0.00', f"${cost_per_1k:.2f}" if costs else 'N/A'],
    'Explainability': ['High', 'None', 'Low'],
    'Offline Capable': ['Yes', 'Yes', 'No']
}
df_summary = pd.DataFrame(summary_data)
df_summary.to_csv(OUTPUT_PATH / 'model_comparison_table.csv', index=False)
print("✅ model_comparison_table.csv")
print("\n" + df_summary.to_string(index=False))

# ============================================================================
# SAVE RAW DATA
# ============================================================================

print("\nSaving raw results...")
with open(OUTPUT_PATH / 'comparison_results.json', 'w') as f:
    json.dump({
        'n_samples': N_SAMPLES,
        'avg_times': {k: float(np.mean(v)) for k, v in times.items()},
        'total_cost': float(sum(costs)) if costs else 0,
        'cost_per_1k': float(cost_per_1k) if costs else 0
    }, f, indent=2)
print("✅ comparison_results.json")

print("\n" + "="*80)
print("✅ EVALUATION COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: {OUTPUT_PATH}")
print("\nGenerated:")
print("  📊 4 visualizations (PNG)")
print("  📋 1 comparison table (CSV)")
print("  💾 1 raw data file (JSON)")
print("\n" + "="*80)
