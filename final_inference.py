"""
ShifaMind: Standalone Inference Module

Provides a simple interface for making predictions with a trained ShifaMind model.
Can be used as a Python module or as a command-line tool.

Usage as module:
    from final_inference import ShifaMindPredictor

    predictor = ShifaMindPredictor(checkpoint_path='path/to/checkpoint.pt')
    result = predictor.predict("Patient with fever and cough...")
    print(result['diagnosis']['name'])

Usage as CLI:
    python final_inference.py --text "Patient with fever..." --output json
    python final_inference.py --file note.txt --format detailed

Author: Mohammed Sameer Syed
Institution: University of Arizona
Project: M.S. in Artificial Intelligence Capstone
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import logging

# Import configuration
from config import (
    CHECKPOINT_FINAL,
    TARGET_CODES,
    ICD_DESCRIPTIONS,
    MAX_SEQUENCE_LENGTH,
    CONCEPT_SCORE_THRESHOLD,
    MAX_CONCEPTS_OUTPUT,
    get_device
)

# Import filtering
from final_concept_filter import get_filtered_top_concepts

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# MODEL ARCHITECTURE (minimal - just for inference)
# ============================================================================

class EnhancedCrossAttention(nn.Module):
    """Cross-attention module for concept fusion"""

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
    """ShifaMind model for diagnosis prediction"""

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

    def forward(self, input_ids, attention_mask, concept_embeddings):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states
        current_hidden = hidden_states[-1]

        for i, fusion_module in enumerate(self.fusion_modules):
            layer_idx = self.fusion_layers[i]
            layer_hidden = hidden_states[layer_idx]

            fused_hidden, _ = fusion_module(
                layer_hidden, concept_embeddings, attention_mask
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

        return {
            'logits': diagnosis_logits,
            'concept_scores': refined_concept_logits
        }


# ============================================================================
# PREDICTOR CLASS
# ============================================================================

class ShifaMindPredictor:
    """
    ShifaMind predictor for clinical note diagnosis.

    This class handles model loading, text processing, and prediction generation.

    Example:
        predictor = ShifaMindPredictor()
        result = predictor.predict("72-year-old male with fever and cough...")
        print(result['diagnosis']['name'])
    """

    def __init__(self,
                 checkpoint_path: Union[str, Path] = CHECKPOINT_FINAL,
                 device: Optional[str] = None):
        """
        Initialize the predictor.

        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device) if device else get_device()

        logger.info(f"Loading ShifaMind model from {self.checkpoint_path}")
        logger.info(f"Using device: {self.device}")

        # Load checkpoint
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Extract components
        self.concept_embeddings = checkpoint['concept_embeddings'].to(self.device)
        self.num_concepts = checkpoint['num_concepts']
        self.concept_cuis = checkpoint['concept_cuis']
        self.concept_names = checkpoint['concept_names']

        # Build concept store
        self.concept_store = {
            'concepts': {
                cui: {'preferred_name': name, 'semantic_types': []}
                for cui, name in self.concept_names.items()
            },
            'concept_to_idx': {cui: i for i, cui in enumerate(self.concept_cuis)},
            'idx_to_concept': {i: cui for i, cui in enumerate(self.concept_cuis)}
        }

        # Initialize tokenizer and model
        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(self.device)

        self.model = ShifaMindModel(
            base_model=base_model,
            num_concepts=self.num_concepts,
            num_classes=len(TARGET_CODES),
            fusion_layers=[9, 11]
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()

        logger.info("Model loaded successfully")

    def predict(self,
                text: str,
                return_concepts: bool = True,
                top_k_concepts: int = MAX_CONCEPTS_OUTPUT) -> Dict:
        """
        Make a prediction on clinical text.

        Args:
            text: Clinical note text
            return_concepts: Whether to include concept predictions
            top_k_concepts: Number of top concepts to return

        Returns:
            Dict with prediction results:
                - diagnosis: Dict with code, name, confidence
                - all_probabilities: Dict of all diagnosis probabilities
                - concepts: List of predicted concepts (if return_concepts=True)
                - metadata: Dict with prediction metadata
        """
        if not text or len(text.strip()) < 10:
            raise ValueError("Text is too short for meaningful prediction")

        # Tokenize
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors='pt'
        ).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(
                encoding['input_ids'],
                encoding['attention_mask'],
                self.concept_embeddings
            )

            # Diagnosis prediction
            diagnosis_logits = outputs['logits']
            diagnosis_probs = torch.sigmoid(diagnosis_logits).cpu().numpy()[0]
            predicted_label = diagnosis_probs.argmax()
            confidence = diagnosis_probs[predicted_label]
            diagnosis_code = TARGET_CODES[predicted_label]

            # Concept prediction (filtered)
            concepts = []
            if return_concepts:
                concept_scores = torch.sigmoid(outputs['concept_scores']).cpu().numpy()[0]
                concepts = get_filtered_top_concepts(
                    concept_scores,
                    self.concept_store,
                    top_k=top_k_concepts
                )

        # Build result
        result = {
            'diagnosis': {
                'code': diagnosis_code,
                'name': ICD_DESCRIPTIONS[diagnosis_code],
                'confidence': float(confidence)
            },
            'all_probabilities': {
                TARGET_CODES[i]: float(diagnosis_probs[i])
                for i in range(len(TARGET_CODES))
            },
            'metadata': {
                'model_version': 'ShifaMind',
                'timestamp': datetime.now().isoformat(),
                'text_length': len(text),
                'device': str(self.device)
            }
        }

        if return_concepts:
            result['concepts'] = concepts
            result['metadata']['num_concepts'] = len(concepts)

        return result

    def predict_batch(self,
                     texts: List[str],
                     batch_size: int = 16) -> List[Dict]:
        """
        Predict on multiple texts efficiently.

        Args:
            texts: List of clinical note texts
            batch_size: Batch size for processing

        Returns:
            List of prediction result dicts
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            for text in batch_texts:
                result = self.predict(text)
                results.append(result)

        return results


# ============================================================================
# CLI INTERFACE
# ============================================================================

def format_output(result: Dict, format_type: str = 'json') -> str:
    """
    Format prediction result for display.

    Args:
        result: Prediction result dict
        format_type: Output format ('json', 'text', 'detailed')

    Returns:
        Formatted string
    """
    if format_type == 'json':
        return json.dumps(result, indent=2)

    elif format_type == 'text':
        diagnosis = result['diagnosis']
        return f"{diagnosis['code']}: {diagnosis['name']} ({diagnosis['confidence']:.1%})"

    elif format_type == 'detailed':
        diagnosis = result['diagnosis']
        output = []
        output.append("="*70)
        output.append("SHIFAMIND PREDICTION")
        output.append("="*70)
        output.append(f"\nDiagnosis: {diagnosis['name']}")
        output.append(f"Code: {diagnosis['code']}")
        output.append(f"Confidence: {diagnosis['confidence']:.1%}")

        output.append("\nAll Probabilities:")
        for code, prob in result['all_probabilities'].items():
            output.append(f"  {code}: {prob:.1%}")

        if 'concepts' in result:
            output.append(f"\nTop Concepts ({len(result['concepts'])}):")
            for i, concept in enumerate(result['concepts'][:5], 1):
                output.append(f"  {i}. {concept['name']} ({concept['score']:.1%})")

        output.append("\n" + "="*70)
        return "\n".join(output)

    else:
        raise ValueError(f"Unknown format: {format_type}")


def main_cli():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='ShifaMind: Clinical diagnosis prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict from text
  python final_inference.py --text "Patient with fever and cough..." --format detailed

  # Predict from file
  python final_inference.py --file note.txt --output json

  # Save to file
  python final_inference.py --file note.txt --save result.json
        """
    )

    parser.add_argument(
        '--text',
        type=str,
        help='Clinical note text to analyze'
    )

    parser.add_argument(
        '--file',
        type=Path,
        help='Path to file containing clinical note'
    )

    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=CHECKPOINT_FINAL,
        help=f'Path to model checkpoint (default: {CHECKPOINT_FINAL})'
    )

    parser.add_argument(
        '--format',
        choices=['json', 'text', 'detailed'],
        default='detailed',
        help='Output format (default: detailed)'
    )

    parser.add_argument(
        '--save',
        type=Path,
        help='Save output to file'
    )

    parser.add_argument(
        '--no-concepts',
        action='store_true',
        help='Disable concept prediction (faster)'
    )

    # Use parse_known_args() to ignore Jupyter/Colab kernel arguments (e.g., -f)
    args, unknown = parser.parse_known_args()

    # Get input text
    if args.text:
        text = args.text
    elif args.file:
        if not args.file.exists():
            logger.error(f"File not found: {args.file}")
            return
        text = args.file.read_text(encoding='utf-8')
    else:
        parser.error("Either --text or --file must be provided")

    # Initialize predictor
    predictor = ShifaMindPredictor(checkpoint_path=args.checkpoint)

    # Make prediction
    logger.info("Making prediction...")
    result = predictor.predict(
        text,
        return_concepts=not args.no_concepts
    )

    # Format output
    output = format_output(result, format_type=args.format)

    # Display or save
    if args.save:
        args.save.write_text(output, encoding='utf-8')
        logger.info(f"Saved to {args.save}")
    else:
        print(output)


if __name__ == '__main__':
    main_cli()
