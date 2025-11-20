"""
ShifaMind: Clinical Knowledge Base Generator

Generates a structured clinical knowledge base from UMLS and ICD-10 sources.
This knowledge base is used during inference to provide clinical context and
evidence for model predictions.

The generator:
1. Parses UMLS MRCONSO and MRDEF files for medical concepts
2. Extracts ICD-10 code descriptions
3. Filters out veterinary/animal concepts
4. Creates structured knowledge entries for each diagnosis
5. Saves to JSON format for fast retrieval

Author: Mohammed Sameer Syed
Institution: University of Arizona
Project: M.S. in Artificial Intelligence Capstone
Date: November 2025
"""

import json
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm
from typing import Dict, List, Set
import argparse
import logging

# Import configuration
from config import (
    MRCONSO_PATH,
    MRDEF_PATH,
    ICD10_CODES_PATH,
    MODEL_PATH,
    TARGET_CODES,
    ICD_DESCRIPTIONS,
    DIAGNOSIS_KEYWORDS,
    TRUSTED_UMLS_SOURCES
)

# Import filtering functions
from final_concept_filter import filter_knowledge_base_entry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_FILE = MODEL_PATH / 'clinical_knowledge_base.json'

print("="*80)
print("SHIFAMIND: CLINICAL KNOWLEDGE BASE GENERATOR")
print("="*80)
logger.info(f"UMLS Path: {MRCONSO_PATH.parent}")
logger.info(f"ICD-10 Path: {ICD10_CODES_PATH.parent}")
logger.info(f"Output: {OUTPUT_FILE}")

# ============================================================================
# PARSING FUNCTIONS
# ============================================================================

def parse_rrf_line(line: str, delimiter: str = '|') -> List[str]:
    """
    Parse a single line from an RRF (Rich Release Format) file.

    Args:
        line: Line from RRF file
        delimiter: Field delimiter (default: |)

    Returns:
        List of fields
    """
    return line.strip().split(delimiter)


def load_concept_names(mrconso_path: Path, target_keywords: Dict[str, Dict]) -> Dict:
    """
    Load concept names from MRCONSO.RRF with animal filtering.

    MRCONSO contains concept names and synonyms from various sources.
    We search for specific medical terms related to our target diagnoses.

    Args:
        mrconso_path: Path to MRCONSO.RRF file
        target_keywords: Dict mapping diagnosis codes to keyword categories

    Returns:
        Dict mapping CUI -> concept information
    """
    logger.info("="*70)
    logger.info("PARSING MRCONSO.RRF - Concept Names (WITH FILTERING)")
    logger.info("="*70)
    logger.info(f"File: {mrconso_path}")
    logger.info(f"Size: {mrconso_path.stat().st_size / (1024**3):.2f} GB")

    # Flatten all keywords for searching
    all_keywords = set()
    for dx_keywords in target_keywords.values():
        for category_keywords in dx_keywords.values():
            all_keywords.update([k.lower() for k in category_keywords])

    logger.info(f"Searching for {len(all_keywords)} unique medical terms...")
    logger.info("Filtering out animal/veterinary concepts...")

    concepts = {}
    filtered_count = 0

    with open(mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="  Scanning MRCONSO", unit=" lines"):
            fields = parse_rrf_line(line)

            if len(fields) < 15:
                continue

            cui = fields[0]
            lat = fields[1]
            ispref = fields[6]
            sab = fields[11]
            str_text = fields[14]

            # Only English
            if lat != 'ENG':
                continue

            # Only trusted sources
            if sab not in TRUSTED_UMLS_SOURCES:
                continue

            # FILTER: Check for animal content
            if filter_knowledge_base_entry(str_text):
                filtered_count += 1
                continue

            str_lower = str_text.lower()

            # Check if matches any target keyword
            matched = False
            for keyword in all_keywords:
                if keyword in str_lower or str_lower in keyword:
                    matched = True
                    break

            if not matched:
                continue

            # Store concept
            if cui not in concepts:
                concepts[cui] = {
                    'name': str_text,
                    'source': sab,
                    'synonyms': []
                }
            else:
                if str_text not in concepts[cui]['synonyms']:
                    concepts[cui]['synonyms'].append(str_text)

    logger.info(f"  ✅ Found {len(concepts)} relevant concepts")
    logger.info(f"  🚫 Filtered out {filtered_count} animal/veterinary concepts")

    return concepts


def load_concept_definitions(mrdef_path: Path, target_cuis: Set[str]) -> Dict[str, str]:
    """
    Load definitions from MRDEF.RRF with animal filtering.

    Args:
        mrdef_path: Path to MRDEF.RRF file
        target_cuis: Set of CUIs to load definitions for

    Returns:
        Dict mapping CUI -> definition text
    """
    logger.info("="*70)
    logger.info("PARSING MRDEF.RRF - Concept Definitions (WITH FILTERING)")
    logger.info("="*70)
    logger.info(f"File: {mrdef_path}")

    if not mrdef_path.exists():
        logger.warning("  ⚠️  MRDEF.RRF not found, skipping definitions")
        return {}

    logger.info(f"Size: {mrdef_path.stat().st_size / (1024**3):.2f} GB")
    logger.info(f"Looking for definitions for {len(target_cuis)} concepts...")

    definitions = {}
    filtered_count = 0

    with open(mrdef_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="  Scanning MRDEF", unit=" lines"):
            fields = parse_rrf_line(line)

            if len(fields) < 6:
                continue

            cui = fields[0]
            definition = fields[5]

            if cui not in target_cuis:
                continue

            # FILTER: Check for animal content in definition
            if filter_knowledge_base_entry(definition):
                filtered_count += 1
                continue

            if cui not in definitions:
                definitions[cui] = definition

    logger.info(f"  ✅ Found {len(definitions)} definitions")
    logger.info(f"  🚫 Filtered out {filtered_count} animal/veterinary definitions")

    return definitions


def load_icd10_descriptions(icd10_path: Path) -> Dict[str, str]:
    """
    Load ICD-10 code descriptions.

    Args:
        icd10_path: Path to ICD-10 codes file

    Returns:
        Dict mapping ICD-10 code -> description
    """
    logger.info("="*70)
    logger.info("PARSING ICD-10 CODES")
    logger.info("="*70)
    logger.info(f"File: {icd10_path}")

    descriptions = {}

    with open(icd10_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            parts = line.split(None, 1)

            if len(parts) == 2:
                code, description = parts
                descriptions[code] = description

    logger.info(f"  ✅ Loaded {len(descriptions)} ICD-10 codes")

    return descriptions


# ============================================================================
# KNOWLEDGE BASE CONSTRUCTION
# ============================================================================

def create_diagnosis_knowledge_entry(diagnosis_code: str,
                                    icd_descriptions: Dict[str, str],
                                    concepts: Dict,
                                    definitions: Dict[str, str]) -> List[Dict]:
    """
    Create knowledge base entries for a specific diagnosis.

    Generates multiple types of knowledge entries:
    1. Diagnosis description (from ICD-10)
    2. Clinical presentation (symptoms)
    3. Physical examination findings
    4. Diagnostic findings (imaging, tests)
    5. Laboratory findings
    6. UMLS concept definitions

    Args:
        diagnosis_code: ICD-10 code (e.g., "J189")
        icd_descriptions: Dict of ICD-10 code -> description
        concepts: Dict of CUI -> concept info
        definitions: Dict of CUI -> definition text

    Returns:
        List of knowledge entry dicts
    """
    diagnosis_name = icd_descriptions.get(diagnosis_code, diagnosis_code)
    keywords = DIAGNOSIS_KEYWORDS.get(diagnosis_code, {})

    entries = []

    # 1. Diagnosis description (from ICD-10)
    entries.append({
        'type': 'diagnosis_description',
        'text': f"{diagnosis_name} ({diagnosis_code}) is a clinical condition requiring medical diagnosis and treatment.",
        'source': f'ICD-10-CM {diagnosis_code}',
        'keywords': keywords.get('core', []) if isinstance(keywords, dict) else keywords[:5],
        'priority': 10
    })

    # 2. Clinical presentation
    if isinstance(keywords, dict) and 'symptoms' in keywords:
        symptoms_text = f"Common presenting symptoms of {diagnosis_name} include: "
        symptoms_text += ", ".join(keywords['symptoms'][:5]) + "."

        entries.append({
            'type': 'clinical_presentation',
            'text': symptoms_text,
            'source': f'Clinical Knowledge - {diagnosis_code}',
            'keywords': keywords['symptoms'],
            'priority': 9
        })

    # 3. Physical examination findings
    if isinstance(keywords, dict) and 'signs' in keywords:
        signs_text = f"Physical examination findings may reveal: "
        signs_text += ", ".join(keywords['signs'][:5]) + "."

        entries.append({
            'type': 'physical_findings',
            'text': signs_text,
            'source': f'Clinical Knowledge - {diagnosis_code}',
            'keywords': keywords['signs'],
            'priority': 8
        })

    # 4. Diagnostic findings
    if isinstance(keywords, dict) and 'findings' in keywords:
        findings_text = f"Diagnostic imaging and tests may show: "
        findings_text += ", ".join(keywords['findings'][:5]) + "."

        entries.append({
            'type': 'diagnostic_findings',
            'text': findings_text,
            'source': f'Clinical Knowledge - {diagnosis_code}',
            'keywords': keywords['findings'],
            'priority': 7
        })

    # 5. Laboratory findings
    if isinstance(keywords, dict) and 'lab' in keywords:
        lab_text = f"Laboratory abnormalities may include: "
        lab_text += ", ".join(keywords['lab'][:5]) + "."

        entries.append({
            'type': 'laboratory_findings',
            'text': lab_text,
            'source': f'Clinical Knowledge - {diagnosis_code}',
            'keywords': keywords['lab'],
            'priority': 6
        })

    # 6. UMLS concept definitions (FILTERED for human medicine only)
    all_keywords = []
    if isinstance(keywords, dict):
        for category_keywords in keywords.values():
            all_keywords.extend([k.lower() for k in category_keywords])
    else:
        all_keywords = [k.lower() for k in keywords]

    concept_count = 0
    for cui, concept_info in concepts.items():
        if concept_count >= 10:
            break

        concept_name = concept_info['name'].lower()

        # Check if concept matches diagnosis keywords
        matched = False
        for kw in all_keywords:
            if kw in concept_name or concept_name in kw:
                matched = True
                break

        if not matched:
            continue

        # Get definition if available (already filtered for animals)
        definition = definitions.get(cui)

        if not definition:
            # Create generic definition
            definition = f"{concept_info['name']} is a medical concept relevant to {diagnosis_name}."

        # Double-check for animal content (safety check)
        if filter_knowledge_base_entry(definition):
            continue

        # Truncate long definitions
        if len(definition) > 300:
            definition = definition[:297] + "..."

        entries.append({
            'type': 'concept_definition',
            'text': definition,
            'source': f"UMLS {cui} ({concept_info['source']})",
            'keywords': [concept_info['name'].lower()],
            'cui': cui,
            'priority': 5
        })

        concept_count += 1

    # Sort by priority (highest first)
    entries.sort(key=lambda x: x['priority'], reverse=True)

    # Remove priority field from output
    for entry in entries:
        del entry['priority']

    return entries


def build_knowledge_base(target_codes: List[str],
                        icd_descriptions: Dict[str, str],
                        concepts: Dict,
                        definitions: Dict[str, str]) -> Dict:
    """
    Build complete knowledge base for all target diagnoses.

    Args:
        target_codes: List of ICD-10 codes
        icd_descriptions: Dict of ICD-10 descriptions
        concepts: Dict of UMLS concepts
        definitions: Dict of UMLS definitions

    Returns:
        Dict mapping diagnosis code -> list of knowledge entries
    """
    logger.info("="*70)
    logger.info("BUILDING KNOWLEDGE BASE (FILTERED)")
    logger.info("="*70)

    knowledge_base = {}

    for code in target_codes:
        logger.info(f"\n📋 Creating knowledge for {code}: {icd_descriptions.get(code, code)}")

        entries = create_diagnosis_knowledge_entry(
            code, icd_descriptions, concepts, definitions
        )

        knowledge_base[code] = entries

        logger.info(f"  ✅ Generated {len(entries)} knowledge entries (all human medicine)")

    total_entries = sum(len(entries) for entries in knowledge_base.values())
    logger.info(f"\n  ✅ Total knowledge entries: {total_entries}")

    return knowledge_base


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(output_path: Path = OUTPUT_FILE):
    """
    Main knowledge base generation pipeline with filtering.

    Args:
        output_path: Path to save the generated knowledge base JSON
    """
    logger.info("="*80)
    logger.info("STARTING FILTERED KNOWLEDGE BASE GENERATION")
    logger.info("="*80)

    # Validate file paths
    logger.info("\n🔍 Validating file paths...")

    required_files = [
        ('MRCONSO', MRCONSO_PATH),
        ('ICD-10', ICD10_CODES_PATH)
    ]

    for name, path in required_files:
        if not path.exists():
            logger.error(f"  ❌ ERROR: {name} not found at {path}")
            logger.error("  Please ensure UMLS and ICD-10 data are downloaded.")
            return
        logger.info(f"  ✅ {name}: {path.stat().st_size / (1024**2):.1f} MB")

    # Parse ICD-10 codes
    icd_descriptions = load_icd10_descriptions(ICD10_CODES_PATH)

    # Parse UMLS MRCONSO (WITH FILTERING)
    concepts = load_concept_names(MRCONSO_PATH, DIAGNOSIS_KEYWORDS)

    # Parse UMLS MRDEF (WITH FILTERING)
    target_cuis = set(concepts.keys())
    definitions = load_concept_definitions(MRDEF_PATH, target_cuis)

    # Build knowledge base
    knowledge_base = build_knowledge_base(
        TARGET_CODES, icd_descriptions, concepts, definitions
    )

    # Save to JSON
    logger.info(f"\n💾 Saving filtered knowledge base to {output_path}...")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)

    logger.info(f"  ✅ Saved: {output_path}")
    logger.info(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")

    # Print summary
    logger.info("="*80)
    logger.info("FILTERED KNOWLEDGE BASE GENERATION COMPLETE")
    logger.info("="*80)

    logger.info("\n📊 SUMMARY:")
    for code in TARGET_CODES:
        num_entries = len(knowledge_base[code])
        logger.info(f"  {code} ({ICD_DESCRIPTIONS[code]}): {num_entries} entries")

    total_entries = sum(len(entries) for entries in knowledge_base.values())
    logger.info(f"\n  Total entries: {total_entries}")
    logger.info(f"  Average per diagnosis: {total_entries / len(TARGET_CODES):.1f}")
    logger.info(f"\n  🚫 All animal/veterinary content filtered out")

    # Show sample entry
    logger.info("\n📄 SAMPLE ENTRY (J189):")
    sample_entry = knowledge_base['J189'][0]
    logger.info(f"  Type: {sample_entry['type']}")
    logger.info(f"  Text: {sample_entry['text'][:150]}...")
    logger.info(f"  Source: {sample_entry['source']}")
    logger.info(f"  Keywords: {', '.join(sample_entry['keywords'][:5])}")

    logger.info("\n✅ Filtered knowledge base ready for use!")
    logger.info("="*80)


if __name__ == '__main__':
    # Command-line interface
    parser = argparse.ArgumentParser(
        description='Generate clinical knowledge base from UMLS and ICD-10',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python final_knowledge_base_generator.py
  python final_knowledge_base_generator.py --output custom_kb.json
        """
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=OUTPUT_FILE,
        help=f'Output path for knowledge base JSON (default: {OUTPUT_FILE})'
    )

    # Use parse_known_args() to ignore Jupyter/Colab kernel arguments (e.g., -f)
    args, unknown = parser.parse_known_args()

    main(output_path=args.output)
