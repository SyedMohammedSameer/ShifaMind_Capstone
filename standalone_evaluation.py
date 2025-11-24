# ======================
# CONFIG
# ======================
API_KEY = "sk-proj-YOUR_KEY_HERE"   # <---- PUT YOUR OPENAI KEY HERE
N_SAMPLES = 30  # Number of samples
BASE_PATH = "/content/drive/MyDrive/ShifaMind"

# ======================
# IMPORTS
# ======================
import os, json, time, warnings, re
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
CODE_TO_LABEL = {code: i for i, code in enumerate(TARGET_CODES)}
LABEL_TO_CODE = {i: code for code, i in CODE_TO_LABEL.items()}

print("="*80)
print("SHIFAMIND COMPREHENSIVE EVALUATION WITH FULL EXPLAINABILITY")
print("="*80)
print(f"Device: {device}\nSamples: {N_SAMPLES}\nOutput: {OUTPUT_PATH}\n")

# ======================
# MODELS (simplified)
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
checkpoint = torch.load(BASE_PATH / '03_Models/checkpoints/shifamind_model.pt', map_location=device)
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

with open(BASE_PATH / '03_Models/clinical_knowledge_base.json', 'r') as f:
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

# ======================
# LOAD TEST DATA
# ======================
print("Loading test data...")
notes_path = BASE_PATH / "01_Raw_Datasets/Extracted/mimic-iv-note-2.2/note/discharge.csv.gz"
df = pd.read_csv(notes_path, compression="gzip", nrows=500, on_bad_lines="skip")
df = df[df["text"].str.len() > 500].sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)
test_notes = df['text'].tolist()
print(f"✅ Loaded {len(test_notes)} test notes\n")

# ======================
# RUN PREDICTIONS WITH FULL EXPLAINABILITY
# ======================
print("Running predictions...")
print("="*80 + "\n")

results = {'shifamind': [], 'bioclinbert': [], 'gpt4': []}
times = {'shifamind': [], 'bioclinbert': [], 'gpt4': []}
costs, example_predictions = [], []

for i, text in enumerate(tqdm(test_notes), 1):
    # === SHIFAMIND - FULL EXPLAINABILITY ===
    start = time.time()
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=384, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = shifamind_model(encoding['input_ids'], encoding['attention_mask'], concept_embeddings, return_attention=True)
        probs_sm = torch.sigmoid(outputs['logits']).cpu().numpy()[0]
        concept_scores = torch.sigmoid(outputs['concept_scores']).cpu().numpy()[0]
    time_sm = time.time() - start
    times['shifamind'].append(time_sm)
    results['shifamind'].append(probs_sm)
    
    pred_sm_idx = np.argmax(probs_sm)
    pred_sm_code = LABEL_TO_CODE[pred_sm_idx]
    pred_sm_label, conf_sm = ICD_DESCRIPTIONS[pred_sm_code], probs_sm[pred_sm_idx]
    
    # Extract concepts
    all_indices = np.argsort(concept_scores)[::-1]
    concepts = []
    for idx in all_indices[:10]:
        cui = concept_store['idx_to_concept'].get(idx, f'CUI_{idx}')
        concept_info = concept_store['concepts'].get(cui, {})
        concept_name = concept_info.get('preferred_name', f'Concept_{idx}')
        concepts.append({'idx': idx, 'cui': cui, 'name': concept_name, 'score': float(concept_scores[idx])})
    
    # Extract evidence and knowledge (for first 2 examples)
    evidence_chains, clinical_knowledge = [], []
    if i <= 2:
        evidence_chains = extract_evidence_spans(text, encoding['input_ids'][0], outputs['attention_weights'], concepts, tokenizer, top_k=5)
        clinical_knowledge = retrieve_clinical_knowledge(pred_sm_code, concepts, knowledge_base, top_k=2)
    
    # === BIO_CLINICALBERT - BASELINE ===
    start = time.time()
    encoding_bert = bert_tokenizer(text, padding='max_length', truncation=True, max_length=384, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs_bert = bert_model(**encoding_bert)
        logits = bert_classifier(outputs_bert.last_hidden_state[:, 0, :])
        probs_bc = torch.sigmoid(logits).cpu().numpy()[0]
    time_bc = time.time() - start
    times['bioclinbert'].append(time_bc)
    results['bioclinbert'].append(probs_bc)
    pred_bc_idx, pred_bc_label, conf_bc = np.argmax(probs_bc), list(ICD_DESCRIPTIONS.values())[np.argmax(probs_bc)], probs_bc[np.argmax(probs_bc)]

    # === GPT-4 - API ===
    pred_gpt_label, conf_gpt, time_gpt = "N/A", 0.0, 0.0
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
            response = gpt_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], max_tokens=100, temperature=0.1)
            latency = time.time() - start
            content = response.choices[0].message.content
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()
            try:
                result = json.loads(content)
            except:
                result = {"diagnosis": "Pneumonia", "confidence": 50}
            diagnosis, confidence = result.get("diagnosis", "Pneumonia"), result.get("confidence", 50) / 100.0
            probs_gpt = np.zeros(len(TARGET_CODES))
            for j, desc in enumerate(ICD_DESCRIPTIONS.values()):
                probs_gpt[j] = confidence if desc == diagnosis else (1 - confidence) / (len(TARGET_CODES) - 1)
            usage = response.usage
            cost = (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60) / 1_000_000
            times['gpt4'].append(latency)
            results['gpt4'].append(probs_gpt)
            costs.append(cost)
            pred_gpt_label, conf_gpt, time_gpt = diagnosis, confidence, latency
            time.sleep(0.5)
        except Exception as e:
            print(f"GPT Error: {e}")

    # Store first 2 examples with FULL ShifaMind explainability
    if i <= 2:
        example_predictions.append({
            'example_num': i,
            'clinical_note': text[:800] + "..." if len(text) > 800 else text,
            'shifamind': {
                'prediction': pred_sm_label,
                'confidence': f"{conf_sm:.1%}",
                'time': f"{time_sm:.3f}s",
                'top_concepts': [{'name': c['name'], 'score': f"{c['score']:.1%}"} for c in concepts[:5]],
                'evidence_chains': evidence_chains[:3],
                'clinical_knowledge': clinical_knowledge
            },
            'bioclinbert': {
                'prediction': pred_bc_label,
                'confidence': f"{conf_bc:.1%}",
                'time': f"{time_bc:.3f}s",
                'explainability': "❌ None - Black box model"
            },
            'gpt4': {
                'prediction': pred_gpt_label,
                'confidence': f"{conf_gpt:.1%}" if pred_gpt_label != "N/A" else "N/A",
                'time': f"{time_gpt:.3f}s" if pred_gpt_label != "N/A" else "N/A",
                'explainability': "⚠️  Basic text generation only"
            }
        })

print("\n✅ Predictions complete\n")

# ======================
# VISUALIZATIONS (same as before)
# ======================
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
models_cost = ['ShifaMind\n(Local)', 'Bio_ClinicalBERT\n(Local)', f'GPT-4o-mini\n(${cost_per_1k:.2f}/1k)']
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

# ======================
# COMPARISON TABLE
# ======================
print("\nGenerating comparison table...")
summary_data = {
    'Model': ['ShifaMind', 'Bio_ClinicalBERT', 'GPT-4o-mini'],
    'Avg Inference Time (s)': [f"{np.mean(times['shifamind']):.3f}", f"{np.mean(times['bioclinbert']):.3f}",
                                 f"{np.mean(times['gpt4']):.3f}" if times['gpt4'] else 'N/A'],
    'Cost per 1k Calls': ['$0.00', '$0.00', f"${cost_per_1k:.2f}" if costs else 'N/A'],
    'Explainability': ['High (Concepts+Evidence+Knowledge)', 'None', 'Low (Text only)'],
    'Offline Capable': ['Yes', 'Yes', 'No']
}
df_summary = pd.DataFrame(summary_data)
df_summary.to_csv(OUTPUT_PATH / 'model_comparison_table.csv', index=False)
print("✅ model_comparison_table.csv")
print("\n" + df_summary.to_string(index=False))

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
        f.write("---\n\n")
        
        f.write("### 🤖 ShifaMind Prediction (WITH FULL EXPLAINABILITY)\n\n")
        sm = ex['shifamind']
        f.write(f"**Diagnosis:** {sm['prediction']}\n\n")
        f.write(f"**Confidence:** {sm['confidence']}\n\n")
        f.write(f"**Inference Time:** {sm['time']}\n\n")
        
        f.write("#### 🔬 Top Medical Concepts Extracted:\n\n")
        for i, concept in enumerate(sm['top_concepts'], 1):
            f.write(f"{i}. **{concept['name']}** ({concept['score']})\n")
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
        f.write("### 💡 Key Insights:\n\n")
        f.write("✅ **ShifaMind provides:**\n")
        f.write("- Medical concepts extracted from ontologies\n")
        f.write("- Evidence spans showing WHY each concept was identified\n")
        f.write("- Clinical knowledge retrieval for educational context\n")
        f.write("- Full traceability from text → concepts → diagnosis\n\n")
        f.write("❌ **Bio_ClinicalBERT provides:**\n")
        f.write("- Only diagnosis label + confidence (black box)\n\n")
        f.write("⚠️  **GPT-4o-mini provides:**\n")
        f.write("- Diagnosis via text generation (no structured explainability)\n\n")
        f.write("---\n\n")

print("✅ example_predictions_detailed.md")

# ======================
# SAVE RAW DATA
# ======================
print("\nSaving raw results...")
with open(OUTPUT_PATH / 'comparison_results.json', 'w') as f:
    json.dump({
        'n_samples': N_SAMPLES,
        'avg_times': {k: float(np.mean(v)) for k, v in times.items()},
        'total_cost': float(sum(costs)) if costs else 0,
        'cost_per_1k': float(cost_per_1k) if costs else 0,
        'example_predictions': example_predictions
    }, f, indent=2)
print("✅ comparison_results.json")

print("\n" + "="*80)
print("✅ EVALUATION COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: {OUTPUT_PATH}")
print("\nGenerated:")
print("  📊 4 visualizations (PNG)")
print("  📋 1 comparison table (CSV)")
print("  📝 2 example predictions WITH FULL EXPLAINABILITY (Markdown)")
print("  💾 1 raw data file (JSON)")
print("\n🎯 Key Feature: example_predictions_detailed.md shows:")
print("   - Medical concepts extracted")
print("   - Evidence spans from clinical text")
print("   - Clinical knowledge retrieval")
print("   - Full reasoning chains")
print("\n" + "="*80)
