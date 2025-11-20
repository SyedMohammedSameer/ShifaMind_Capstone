#!/usr/bin/env python3
"""
ShifaMind: Interactive Demo with FILTERED Concepts (V2)

UPDATED: Removes animal/veterinary concepts from predictions

Author: Mohammed Sameer Syed
Date: November 2025
Author: Mohammed Sameer Syed
Institution: University of Arizona
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import gradio as gr
import math
import re

# ============================================================================
# CONCEPT FILTERING (Remove animal/veterinary concepts)
# ============================================================================

ANIMAL_KEYWORDS = [
    'cattle', 'bovine', 'cow', 'cows', 'bull', 'bulls',
    'pig', 'pigs', 'swine', 'porcine', 'hog', 'hogs',
    'horse', 'horses', 'equine', 'foal',
    'sheep', 'ovine', 'lamb',
    'goat', 'goats', 'caprine',
    'dog', 'dogs', 'canine', 'puppy',
    'cat', 'cats', 'feline', 'kitten',
    'chicken', 'poultry', 'avian', 'bird',
    'fish', 'fishes', 'aquatic',
    'rodent', 'mouse', 'mice', 'rat', 'rats',
    'veterinary', 'veterinarian', 'animal', 'animals',
    'livestock', 'farm animal', 'domestic animal',
    'wildlife', 'zoo', 'zoological'
]

EXCLUSION_PATTERNS = [
    r'\bof cattle\b', r'\bof pigs\b', r'\bof swine\b',
    r'\bin cattle\b', r'\bin pigs\b', r'\bin swine\b',
    r'\bANIMAL\b', r'\bVETERINARY\b'
]

def is_animal_concept(concept_name):
    """Check if concept is animal/veterinary related"""
    concept_lower = concept_name.lower()
    
    for keyword in ANIMAL_KEYWORDS:
        if keyword in concept_lower:
            return True
    
    for pattern in EXCLUSION_PATTERNS:
        if re.search(pattern, concept_name, re.IGNORECASE):
            return True
    
    return False

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
CHECKPOINT_PATH = BASE_PATH / '03_Models/checkpoints/shifamind_model.pt'
KB_PATH = BASE_PATH / '03_Models/clinical_knowledge_base.json'

TARGET_CODES = ['J189', 'I5023', 'A419', 'K8000']
CODE_TO_LABEL = {code: i for i, code in enumerate(TARGET_CODES)}
LABEL_TO_CODE = {i: code for code, i in CODE_TO_LABEL.items()}

ICD_DESCRIPTIONS = {
    'J189': 'Pneumonia, unspecified organism',
    'I5023': 'Acute on chronic systolic heart failure',
    'A419': 'Sepsis, unspecified organism',
    'K8000': 'Calculus of gallbladder with acute cholecystitis'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("🏥 SHIFAMIND: LIVE DEMO (FILTERED)")
print("="*80)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class EnhancedCrossAttention(nn.Module):
    """Cross-attention with attention weight capture"""

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

    def forward(self, hidden_states, concept_embeddings, attention_mask=None, return_attention=False):
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_concepts = concept_embeddings.shape[0]

        concepts_batch = concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights_dropout = self.dropout(attention_weights)

        context = torch.matmul(attention_weights_dropout, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        gate_input = torch.cat([hidden_states, context], dim=-1)
        gate_values = torch.sigmoid(self.gate(gate_input))
        output = hidden_states + gate_values * context
        output = self.layer_norm(output)

        if return_attention:
            return output, attention_weights
        return output


class ShifaMindModel(nn.Module):
    """ShifaMind Phase 2 model"""

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

    def forward(self, input_ids, attention_mask, concept_embeddings, return_diagnosis_only=False, return_attention=False):
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

        fusion_attentions = {}

        for i, fusion_module in enumerate(self.fusion_modules):
            layer_idx = self.fusion_layers[i]
            layer_hidden = hidden_states[layer_idx]

            if return_attention:
                fused_hidden, attn_weights = fusion_module(
                    layer_hidden, concept_embeddings, attention_mask, return_attention=True
                )
                fusion_attentions[f'layer_{layer_idx}'] = attn_weights
            else:
                fused_hidden = fusion_module(
                    layer_hidden, concept_embeddings, attention_mask, return_attention=False
                )

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

        result = {
            'logits': diagnosis_logits,
            'concept_scores': refined_concept_logits,
        }

        if return_attention:
            fusion_attentions['input_ids'] = input_ids
            result['attention_weights'] = fusion_attentions

        return result


# ============================================================================
# EVIDENCE EXTRACTION & KNOWLEDGE RETRIEVAL
# ============================================================================

def extract_evidence_spans(text, input_ids, attention_weights, concepts, tokenizer, top_k=5, span_window=10):
    """Extract evidence text spans using attention weights"""

    layer_9_attn = attention_weights.get('layer_9')
    layer_11_attn = attention_weights.get('layer_11')

    if layer_9_attn is None or layer_11_attn is None:
        return []

    attn_9 = layer_9_attn.squeeze(0).mean(0)
    attn_11 = layer_11_attn.squeeze(0).mean(0)
    avg_attention = (attn_9 + attn_11) / 2

    tokens = tokenizer.convert_ids_to_tokens(input_ids.cpu().tolist())

    evidence_chains = []

    for concept in concepts[:top_k]:
        concept_idx = concept.get('idx', 0)

        if concept_idx >= avg_attention.shape[1]:
            continue

        concept_attention = avg_attention[:, concept_idx]

        topk_values, topk_indices = torch.topk(
            concept_attention,
            k=min(5, len(tokens))
        )

        spans = []
        for token_idx in topk_indices:
            token_idx = token_idx.item()

            start = max(0, token_idx - span_window)
            end = min(len(tokens), token_idx + span_window + 1)

            span_tokens = tokens[start:end]
            span_text = tokenizer.convert_tokens_to_string(span_tokens)

            span_text = span_text.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').strip()
            span_text = re.sub(r'\s+', ' ', span_text)
            span_text = span_text.strip('.,;: ')

            if len(span_text) > 20:
                spans.append(span_text)

        unique_spans = []
        seen = set()
        for span in spans:
            span_lower = span.lower()
            if span_lower not in seen:
                unique_spans.append(span)
                seen.add(span_lower)
                if len(unique_spans) >= 3:
                    break

        evidence_chains.append({
            'concept': concept['name'],
            'cui': concept.get('cui', 'UNKNOWN'),
            'score': float(concept['score']),
            'evidence_spans': unique_spans[:3]
        })

    return evidence_chains


def retrieve_clinical_knowledge(diagnosis, concepts, knowledge_base, top_k=3):
    """Retrieve relevant clinical knowledge"""

    if diagnosis not in knowledge_base:
        return []

    all_entries = knowledge_base[diagnosis]

    concept_keywords = set()
    for c in concepts[:10]:
        concept_keywords.update(c['name'].lower().split())

    scored_entries = []
    for entry in all_entries:
        entry_words = set(entry['text'].lower().split())
        overlap = len(concept_keywords & entry_words)

        keyword_bonus = 0
        if 'keywords' in entry:
            for kw in entry['keywords']:
                if kw.lower() in concept_keywords:
                    keyword_bonus += 2

        type_bonus = {
            'clinical_presentation': 3,
            'diagnosis_description': 2,
            'physical_findings': 2,
            'diagnostic_findings': 2,
            'concept_definition': 1
        }.get(entry['type'], 0)

        total_score = overlap + keyword_bonus + type_bonus
        scored_entries.append((total_score, entry))

    scored_entries.sort(reverse=True, key=lambda x: x[0])

    return [
        {
            'text': entry['text'],
            'source': entry['source'],
            'type': entry['type'],
            'relevance_score': score
        }
        for score, entry in scored_entries[:top_k]
        if score > 0
    ]


# ============================================================================
# LOAD MODEL & KNOWLEDGE BASE
# ============================================================================

print(f"\n📂 Loading checkpoint...")

if not CHECKPOINT_PATH.exists():
    print(f"❌ ERROR: Checkpoint not found")
    sys.exit(1)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

concept_embeddings = checkpoint['concept_embeddings'].to(device)
num_concepts = checkpoint['num_concepts']
concept_cuis = checkpoint['concept_cuis']
concept_names = checkpoint['concept_names']

print(f"  ✅ Loaded {num_concepts} concepts")

concept_store = {
    'concepts': {
        cui: {'preferred_name': name, 'semantic_types': []}
        for cui, name in concept_names.items()
    },
    'concept_to_idx': {cui: i for i, cui in enumerate(concept_cuis)},
    'idx_to_concept': {i: cui for i, cui in enumerate(concept_cuis)}
}

print(f"\n📦 Loading Bio_ClinicalBERT...")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

model = ShifaMindModel(
    base_model=base_model,
    num_concepts=num_concepts,
    num_classes=len(TARGET_CODES),
    fusion_layers=[9, 11]
).to(device)

model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

print(f"  ✅ Model ready")

print(f"\n📚 Loading knowledge base...")

if not KB_PATH.exists():
    print(f"❌ ERROR: Knowledge base not found")
    sys.exit(1)

with open(KB_PATH, 'r') as f:
    knowledge_base = json.load(f)

print(f"  ✅ Knowledge base loaded")
print("\n✅ All components ready! (WITH ANIMAL FILTERING)")
print("="*80 + "\n")

# ============================================================================
# PREDICTION FUNCTION (WITH FILTERING)
# ============================================================================

def predict_with_evidence(clinical_note):
    """Generate prediction with FILTERED concepts (no animals)"""

    if not clinical_note or len(clinical_note.strip()) < 20:
        return (
            "⚠️ Please enter a valid clinical note",
            "", "", "", "{}"
        )

    try:
        encoding = tokenizer(
            clinical_note,
            padding='max_length',
            truncation=True,
            max_length=384,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model(
                encoding['input_ids'],
                encoding['attention_mask'],
                concept_embeddings,
                return_attention=True
            )

            diagnosis_logits = outputs['logits']
            diagnosis_probs = torch.sigmoid(diagnosis_logits).cpu().numpy()[0]
            predicted_label = diagnosis_probs.argmax()
            confidence = diagnosis_probs[predicted_label]
            diagnosis_code = LABEL_TO_CODE[predicted_label]
            diagnosis_name = ICD_DESCRIPTIONS[diagnosis_code]

            concept_scores = torch.sigmoid(outputs['concept_scores']).cpu().numpy()[0]
            all_indices = np.argsort(concept_scores)[::-1]

            # FILTER: Get only human medical concepts
            concepts = []
            filtered_count = 0

            for idx in all_indices:
                if len(concepts) >= 10:
                    break

                cui = concept_store['idx_to_concept'].get(idx, f'CUI_{idx}')
                concept_info = concept_store['concepts'].get(cui, {})
                concept_name = concept_info.get('preferred_name', f'Concept_{idx}')

                # Check if animal concept
                if is_animal_concept(concept_name):
                    filtered_count += 1
                    continue

                concepts.append({
                    'idx': idx,
                    'cui': cui,
                    'name': concept_name,
                    'score': float(concept_scores[idx]),
                    'semantic_types': concept_info.get('semantic_types', [])
                })

            if filtered_count > 0:
                print(f"  🚫 Filtered {filtered_count} animal/veterinary concepts")

            # Extract evidence
            evidence = extract_evidence_spans(
                text=clinical_note,
                input_ids=encoding['input_ids'][0],
                attention_weights=outputs['attention_weights'],
                concepts=concepts,
                tokenizer=tokenizer,
                top_k=5
            )

            # Retrieve knowledge
            clinical_knowledge = retrieve_clinical_knowledge(
                diagnosis=diagnosis_code,
                concepts=concepts,
                knowledge_base=knowledge_base,
                top_k=3
            )

        # Format outputs
        diagnosis_text = f"""## 🎯 {diagnosis_name}

**Code:** {diagnosis_code}  
**Confidence:** {confidence:.1%}
"""

        confidence_html = "### All Diagnosis Probabilities:\n\n"
        for i, code in enumerate(TARGET_CODES):
            prob = diagnosis_probs[i]
            bar_length = int(prob * 20)
            bar = "█" * bar_length
            confidence_html += f"**{code}** ({ICD_DESCRIPTIONS[code][:30]}...): {prob:.1%} {bar}\n\n"

        evidence_html = "## 📋 Evidence Chains\n\n"
        if evidence:
            for i, item in enumerate(evidence[:5], 1):
                evidence_html += f"### {i}. {item['concept']} ({item['score']:.1%})\n\n"
                if item['evidence_spans']:
                    for span in item['evidence_spans']:
                        evidence_html += f'> "{span}"\n\n'
                else:
                    evidence_html += "> *No direct evidence found*\n\n"
        else:
            evidence_html += "*No evidence extracted*\n"

        knowledge_html = "## 📚 Clinical Knowledge\n\n"
        if clinical_knowledge:
            for i, entry in enumerate(clinical_knowledge, 1):
                knowledge_html += f"### {i}. {entry['type'].replace('_', ' ').title()}\n\n"
                knowledge_html += f"{entry['text']}\n\n"
                knowledge_html += f"*Source: {entry['source']} (relevance: {entry['relevance_score']})*\n\n"
        else:
            knowledge_html += "*No knowledge retrieved*\n"

        json_output = {
            'diagnosis': {
                'code': diagnosis_code,
                'name': diagnosis_name,
                'confidence': float(confidence)
            },
            'reasoning_chain': evidence,
            'clinical_knowledge': clinical_knowledge,
            'all_probabilities': {
                ICD_DESCRIPTIONS[LABEL_TO_CODE[i]]: float(diagnosis_probs[i])
                for i in range(len(diagnosis_probs))
            },
            'metadata': {
                'model_version': 'ShifaMind',
                'timestamp': datetime.now().isoformat(),
                'note_length': len(clinical_note),
                'num_concepts': len(concepts),
                'filtered_animal_concepts': filtered_count,
                'num_evidence_chains': len(evidence),
                'num_knowledge_entries': len(clinical_knowledge)
            }
        }

        return (
            diagnosis_text,
            confidence_html,
            evidence_html,
            knowledge_html,
            json.dumps(json_output, indent=2)
        )

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return (error_msg, "", "", "", "{}")


# ============================================================================
# EXAMPLE CASES
# ============================================================================

EXAMPLE_CASES = [
    # PNEUMONIA - 3 cases
    ["""72-year-old male presents with 3-day history of fever (38.9°C), productive cough with yellow-green sputum, and progressive shortness of breath. Reports pleuritic chest pain worse with deep inspiration. Physical examination reveals decreased breath sounds and crackles in the right lower lobe. Respiratory rate 24/min, oxygen saturation 91% on room air. Chest X-ray demonstrates right lower lobe infiltrate. White blood cell count elevated at 15,200."""],
    ["""85-year-old female nursing home resident with altered mental status and fever for 2 days. Temperature 39.1°C, heart rate 110 bpm. Examination shows coarse crackles bilaterally in lung bases. Patient appears lethargic and confused. Chest radiograph reveals bilateral lower lobe opacities. Sputum culture pending. Elevated WBC at 16,800 with left shift."""],
    ["""55-year-old male smoker with COPD presents with worsening dyspnea and increased sputum production. Fever 38.5°C, tachypneic at 28 breaths/min. Auscultation reveals diminished breath sounds and bronchial breathing in left upper lobe. CXR shows left upper lobe consolidation. Labs notable for leukocytosis 14,500 and elevated procalcitonin."""],

    # HEART FAILURE - 3 cases
    ["""68-year-old male with history of ischemic cardiomyopathy presents with 2-week worsening dyspnea on exertion and orthopnea requiring 3 pillows to sleep. Reports bilateral lower extremity edema and 10-pound weight gain. Physical exam reveals elevated JVP at 12 cm, bibasilar crackles, S3 gallop, and 3+ pitting edema to knees bilaterally. BNP elevated at 1,450 pg/mL. Chest X-ray shows pulmonary vascular congestion and bilateral pleural effusions. Echocardiogram demonstrates EF 25%."""],
    ["""75-year-old female with known CHF presents with acute decompensation. Patient reports paroxysmal nocturnal dyspnea and orthopnea. Unable to lie flat. Physical examination notable for respiratory distress, bibasilar crackles, jugular venous distension, and S3 heart sound. Lower extremity edema 3+. BNP 2,100. CXR reveals cardiomegaly with pulmonary edema."""],
    ["""62-year-old male with dilated cardiomyopathy admitted with acute on chronic systolic heart failure exacerbation. Reports increasing shortness of breath, fatigue, and decreased exercise tolerance over past month. Weight gain of 15 pounds. Exam shows elevated JVP, hepatomegaly, ascites, and bilateral lower extremity edema. Echo shows severely reduced EF at 20%. NT-proBNP 3,500."""],

    # SEPSIS - 3 cases
    ["""45-year-old female with known UTI presents to ED with altered mental status, fever 39.8°C, and hypotension (BP 85/50). Patient appears confused and diaphoretic. Heart rate 125 bpm, respiratory rate 32/min. Skin is warm and flushed. Laboratory findings show WBC 22,000 with bandemia, lactate 4.5 mmol/L, creatinine elevated at 2.1 from baseline 0.8. Blood cultures drawn. Meeting sepsis criteria with suspected urosepsis."""],
    ["""70-year-old male post-operative day 3 from abdominal surgery develops fever 38.9°C, tachycardia 130 bpm, hypotension 80/45 mmHg requiring vasopressors. Patient is lethargic with decreased urine output. Labs remarkable for WBC 24,500, lactate 5.2, and acute kidney injury. Procalcitonin elevated at 15. Suspected intra-abdominal infection with septic shock."""],
    ["""82-year-old female with pneumonia progressing to sepsis. Temperature 40.1°C, BP 78/40, HR 140, RR 35. Patient is confused and diaphoretic. Skin mottled with delayed capillary refill. Labs show WBC 28,000, lactate 6.8, platelets dropping to 90,000. Creatinine 2.8, liver enzymes elevated. Multiple organ dysfunction. Blood cultures positive for Streptococcus pneumoniae."""],

    # CHOLECYSTITIS - 3 cases
    ["""52-year-old obese female presents with 6-hour history of severe right upper quadrant pain radiating to right shoulder. Pain began after fatty meal. Associated with nausea and vomiting. Temperature 38.4°C. Physical examination reveals positive Murphy's sign with inspiratory arrest on palpation of RUQ. Abdominal ultrasound shows gallbladder wall thickening 5mm, pericholecystic fluid, and multiple gallstones. WBC 13,500."""],
    ["""65-year-old male with sudden onset severe epigastric and right upper quadrant abdominal pain for 8 hours. Reports fever, chills, and vomiting. Examination shows marked RUQ tenderness with guarding and positive Murphy's sign. Temperature 39.2°C. CT abdomen reveals distended gallbladder with wall thickening, gallstones, and surrounding fluid. Labs show leukocytosis 16,200, elevated alkaline phosphatase and mild hyperbilirubinemia."""],
    ["""48-year-old female with history of gallstones presents with acute cholecystitis. Severe RUQ pain for 12 hours, worse with movement. Fever 38.8°C, positive Murphy's sign, mild jaundice noted. Ultrasound demonstrates thick-walled gallbladder (6mm) with stones and pericholecystic fluid. WBC 15,800 with left shift. Elevated alkaline phosphatase 250 U/L and total bilirubin 3.2 mg/dL."""],
]

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

with gr.Blocks(
    title="ShifaMind - Filtered",
    theme=gr.themes.Soft()
) as demo:
    
    gr.Markdown("""
    # 🏥 ShifaMind: Clinical AI with Evidence & Knowledge
    
    ## ✨ FILTERED VERSION - Human Medicine Only ✨
    
    This system provides **explainable diagnosis predictions** by:
    - 🎯 **Diagnosing** clinical conditions from unstructured notes
    - 📋 **Extracting evidence** spans that support each medical concept
    - 📚 **Retrieving clinical knowledge** from UMLS and ICD-10
    - 🚫 **Filtering out** veterinary/animal concepts (human medicine only)
    
    **Supported Diagnoses:**
    - **J189**: Pneumonia, unspecified organism
    - **I5023**: Acute on chronic systolic heart failure  
    - **A419**: Sepsis, unspecified organism
    - **K8000**: Calculus of gallbladder with acute cholecystitis
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="📝 Clinical Note",
                placeholder="Enter clinical note...",
                lines=12
            )
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                submit_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            diagnosis_output = gr.Markdown(label="🎯 Diagnosis")
            confidence_output = gr.Markdown(label="📊 Confidence")
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1):
            evidence_output = gr.Markdown(label="📋 Evidence Chains")
        
        with gr.Column(scale=1):
            knowledge_output = gr.Markdown(label="📚 Clinical Knowledge")
    
    gr.Markdown("---")
    
    with gr.Row():
        json_output = gr.Code(label="📄 JSON Output", language="json", lines=15)
    
    gr.Markdown("---\n## 📚 Example Cases (Click to load)")
    
    gr.Examples(
        examples=EXAMPLE_CASES,
        inputs=input_text
    )
    
    submit_btn.click(
        fn=predict_with_evidence,
        inputs=[input_text],
        outputs=[diagnosis_output, confidence_output, evidence_output, knowledge_output, json_output]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", "", "", "{}"),
        inputs=[],
        outputs=[diagnosis_output, confidence_output, evidence_output, knowledge_output, json_output]
    )
    
    gr.Markdown("""
    ---
    **Model**: ShifaMind | **Performance**: F1 0.8010, AUROC 0.9122 | **Concepts**: Human medicine only  
    **Author**: Mohammed Sameer Syed | **Institution**: University of Arizona
    """)

if __name__ == '__main__':
    print("\n🚀 LAUNCHING FILTERED DEMO")
    print("="*80 + "\n")
    
    demo.launch(share=True, server_port=None, show_error=True)
