"""
ShifaMind: Concept Filtering Module

Post-processes predicted concepts to remove veterinary/animal content.
This module is applied at inference time and doesn't require retraining.

The filter ensures that only human medical concepts are presented to users,
removing any veterinary, animal-related, or research-only concepts that may
have been included in the UMLS knowledge base.

Author: Mohammed Sameer Syed
Institution: University of Arizona
Project: M.S. in Artificial Intelligence Capstone
Date: November 2025
"""

import re
from typing import List, Dict, Optional, Set
import numpy as np

# Import configuration
from config import (
    ANIMAL_KEYWORDS,
    EXCLUSION_PATTERNS,
    EXCLUDED_SEMANTIC_TYPES,
    EXCLUDED_RESEARCH_TYPES
)

# ============================================================================
# CORE FILTERING FUNCTIONS
# ============================================================================

def is_animal_concept(concept_name: str, semantic_types: Optional[List[str]] = None) -> bool:
    """
    Check if a concept is animal/veterinary related.

    This function applies multiple filters to identify non-human medical concepts:
    1. Keyword matching (e.g., "cattle", "swine", "veterinary")
    2. Regex pattern matching (e.g., "of cattle", "in pigs")
    3. Semantic type filtering (e.g., "Mammal", "Animal")

    Args:
        concept_name: String name of the concept (e.g., "Pneumonia in cattle")
        semantic_types: Optional list of UMLS semantic types (e.g., ["Disease or Syndrome"])

    Returns:
        bool: True if concept should be filtered out (is animal/veterinary), False otherwise

    Examples:
        >>> is_animal_concept("Pneumonia")
        False
        >>> is_animal_concept("Pneumonia in cattle")
        True
        >>> is_animal_concept("African Swine Fever")
        True
        >>> is_animal_concept("Heart failure", ["Disease or Syndrome"])
        False
    """
    concept_lower = concept_name.lower()

    # Check for animal keywords
    for keyword in ANIMAL_KEYWORDS:
        if keyword in concept_lower:
            return True

    # Check for exclusion patterns
    for pattern in EXCLUSION_PATTERNS:
        if re.search(pattern, concept_name, re.IGNORECASE):
            return True

    # Check semantic types if provided
    if semantic_types:
        for sem_type in semantic_types:
            if sem_type in EXCLUDED_SEMANTIC_TYPES:
                return True
            if sem_type in EXCLUDED_RESEARCH_TYPES:
                return True

    return False


def filter_concepts(concepts: List[Dict],
                   keep_top_n: int = 10,
                   min_human_concepts: int = 5) -> List[Dict]:
    """
    Filter out animal/veterinary concepts from predictions.

    Processes a list of predicted concepts and removes any that are identified
    as animal or veterinary-related, keeping only human medical concepts.

    Args:
        concepts: List of concept dicts with keys: 'name', 'score', 'semantic_types', etc.
        keep_top_n: Maximum number of concepts to keep after filtering
        min_human_concepts: Minimum human concepts to return (for monitoring)

    Returns:
        List of filtered concepts (human medicine only), sorted by score

    Examples:
        >>> concepts = [
        ...     {'name': 'Pneumonia', 'score': 0.95},
        ...     {'name': 'African Swine Fever', 'score': 0.93},
        ...     {'name': 'Fever', 'score': 0.91}
        ... ]
        >>> filtered = filter_concepts(concepts, keep_top_n=5)
        >>> len(filtered)
        2
        >>> filtered[0]['name']
        'Pneumonia'
    """
    human_concepts = []
    filtered_concepts = []

    for concept in concepts:
        concept_name = concept.get('name', '')
        semantic_types = concept.get('semantic_types', [])

        if is_animal_concept(concept_name, semantic_types):
            filtered_concepts.append(concept['name'])
        else:
            human_concepts.append(concept)

            if len(human_concepts) >= keep_top_n:
                break

    # Warning if we don't have enough human concepts
    if len(human_concepts) < min_human_concepts:
        print(f"  ⚠️  Warning: Only found {len(human_concepts)} human concepts (wanted {min_human_concepts})")
        if filtered_concepts:
            print(f"  🚫 Filtered out: {', '.join(filtered_concepts[:3])}...")

    return human_concepts


def get_filtered_top_concepts(concept_scores: np.ndarray,
                              concept_store: Dict,
                              top_k: int = 10) -> List[Dict]:
    """
    Get top-K concepts with animal filtering applied.

    This is the main function used during inference to extract and filter
    concepts from model predictions.

    Args:
        concept_scores: Numpy array of concept scores from model output (shape: [num_concepts])
        concept_store: Dict with 'concepts', 'idx_to_concept', and 'concept_to_idx' keys
        top_k: Number of human medical concepts to return

    Returns:
        List of filtered concept dicts, each containing:
            - idx: Index in concept store
            - cui: UMLS CUI identifier
            - name: Preferred concept name
            - score: Prediction score (0-1)
            - semantic_types: List of UMLS semantic types

    Example Usage:
        >>> # After model prediction
        >>> outputs = model(input_ids, attention_mask, concept_embeddings)
        >>> concept_scores = torch.sigmoid(outputs['concept_scores']).cpu().numpy()[0]
        >>> filtered_concepts = get_filtered_top_concepts(concept_scores, concept_store, top_k=10)
        >>> for concept in filtered_concepts:
        ...     print(f"{concept['name']}: {concept['score']:.2%}")
    """
    # Get all concepts sorted by score (highest first)
    all_indices = np.argsort(concept_scores)[::-1]

    concepts = []
    filtered_count = 0

    for idx in all_indices:
        if len(concepts) >= top_k:
            break

        cui = concept_store['idx_to_concept'].get(idx, f'CUI_{idx}')
        concept_info = concept_store['concepts'].get(cui, {})

        concept_name = concept_info.get('preferred_name', f'Concept_{idx}')
        semantic_types = concept_info.get('semantic_types', [])

        # Check if animal concept
        if is_animal_concept(concept_name, semantic_types):
            filtered_count += 1
            continue

        concepts.append({
            'idx': int(idx),
            'cui': cui,
            'name': concept_name,
            'score': float(concept_scores[idx]),
            'semantic_types': semantic_types
        })

    if filtered_count > 0:
        print(f"  🚫 Filtered {filtered_count} animal/veterinary concepts")

    return concepts


def filter_knowledge_base_entry(text: str) -> bool:
    """
    Check if a knowledge base entry contains animal/veterinary content.

    Used during knowledge base generation to filter out entries.

    Args:
        text: Text content of the knowledge base entry

    Returns:
        bool: True if entry should be filtered out, False if it should be kept

    Examples:
        >>> filter_knowledge_base_entry("Pneumonia is a lung infection")
        False
        >>> filter_knowledge_base_entry("Pneumonia of cattle is common")
        True
    """
    text_lower = text.lower()

    # Check keywords
    for keyword in ANIMAL_KEYWORDS:
        if keyword in text_lower:
            return True

    # Check patterns
    for pattern in EXCLUSION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_filtering_stats(all_concepts: List[Dict]) -> Dict:
    """
    Get statistics about filtering results.

    Args:
        all_concepts: List of all concepts before filtering

    Returns:
        Dict with filtering statistics:
            - total: Total concepts
            - animal: Number of animal/veterinary concepts
            - human: Number of human medical concepts
            - filter_rate: Percentage filtered
    """
    total = len(all_concepts)
    animal_count = sum(
        1 for c in all_concepts
        if is_animal_concept(c.get('name', ''), c.get('semantic_types', []))
    )
    human_count = total - animal_count

    return {
        'total': total,
        'animal': animal_count,
        'human': human_count,
        'filter_rate': (animal_count / total * 100) if total > 0 else 0
    }


def print_filter_summary(stats: Dict):
    """Print a summary of filtering statistics"""
    print("\n" + "="*70)
    print("CONCEPT FILTERING SUMMARY")
    print("="*70)
    print(f"  Total concepts: {stats['total']}")
    print(f"  Human medical: {stats['human']} ({stats['human']/stats['total']*100:.1f}%)")
    print(f"  Animal/veterinary: {stats['animal']} ({stats['filter_rate']:.1f}%)")
    print("="*70)


# ============================================================================
# TESTING / VALIDATION
# ============================================================================

def validate_filter():
    """
    Validate the filtering logic with known test cases.

    This function tests the filter with a set of known examples to ensure
    it correctly identifies animal vs. human medical concepts.
    """
    test_concepts = [
        {
            'name': 'Pneumonia, Atypical Interstitial, of Cattle',
            'semantic_types': ['Disease or Syndrome'],
            'expected': True  # Should be filtered
        },
        {
            'name': 'Pneumonia',
            'semantic_types': ['Disease or Syndrome'],
            'expected': False  # Should NOT be filtered
        },
        {
            'name': 'Puerperal sepsis',
            'semantic_types': ['Disease or Syndrome'],
            'expected': False  # Should NOT be filtered
        },
        {
            'name': 'African Swine Fever',
            'semantic_types': ['Disease or Syndrome'],
            'expected': True  # Should be filtered
        },
        {
            'name': 'Leukocytosis',
            'semantic_types': ['Finding'],
            'expected': False  # Should NOT be filtered
        },
        {
            'name': 'Fever',
            'semantic_types': ['Sign or Symptom'],
            'expected': False  # Should NOT be filtered
        },
        {
            'name': 'Veterinary pneumonia',
            'semantic_types': ['Disease or Syndrome'],
            'expected': True  # Should be filtered
        }
    ]

    print("="*70)
    print("CONCEPT FILTER VALIDATION")
    print("="*70)

    all_passed = True

    for i, test in enumerate(test_concepts, 1):
        result = is_animal_concept(test['name'], test['semantic_types'])
        passed = result == test['expected']

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n{i}. {test['name']}")
        print(f"   Expected: {'Filter' if test['expected'] else 'Keep'}")
        print(f"   Got: {'Filter' if result else 'Keep'}")
        print(f"   {status}")

        if not passed:
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("✅ All validation tests passed!")
    else:
        print("❌ Some validation tests failed!")
    print("="*70)

    return all_passed


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == '__main__':
    """Test the filtering module"""

    # Run validation
    validate_filter()

    # Test with example concepts
    test_concepts = [
        {
            'name': 'Pneumonia, Atypical Interstitial, of Cattle',
            'score': 0.98,
            'semantic_types': ['Disease or Syndrome']
        },
        {
            'name': 'Pneumonia',
            'score': 0.95,
            'semantic_types': ['Disease or Syndrome']
        },
        {
            'name': 'Puerperal sepsis',
            'score': 0.94,
            'semantic_types': ['Disease or Syndrome']
        },
        {
            'name': 'African Swine Fever',
            'score': 0.93,
            'semantic_types': ['Disease or Syndrome']
        },
        {
            'name': 'Leukocytosis',
            'score': 0.92,
            'semantic_types': ['Finding']
        },
        {
            'name': 'Fever',
            'score': 0.91,
            'semantic_types': ['Sign or Symptom']
        }
    ]

    print("\n" + "="*70)
    print("CONCEPT FILTER TEST")
    print("="*70)

    print("\n📋 Original concepts:")
    for i, c in enumerate(test_concepts, 1):
        print(f"  {i}. {c['name']} ({c['score']:.1%})")

    print("\n🔍 Filtering...")
    filtered = filter_concepts(test_concepts, keep_top_n=5)

    print(f"\n✅ Filtered concepts ({len(filtered)} remaining):")
    for i, c in enumerate(filtered, 1):
        print(f"  {i}. {c['name']} ({c['score']:.1%})")

    # Print statistics
    stats = get_filtering_stats(test_concepts)
    print_filter_summary(stats)
