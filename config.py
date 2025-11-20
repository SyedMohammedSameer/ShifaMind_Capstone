"""
ShifaMind Configuration Module

Centralized configuration for all ShifaMind components.
This module contains all paths, hyperparameters, and settings used across
the training, evaluation, and deployment pipelines.

Author: Mohammed Sameer Syed
Institution: University of Arizona
Project: M.S. in Artificial Intelligence Capstone
Date: November 2025
"""

from pathlib import Path
import os

# ============================================================================
# PATHS CONFIGURATION
# ============================================================================

# Base path - can be overridden via environment variable
BASE_PATH = Path(os.getenv('SHIFAMIND_BASE_PATH', '/content/drive/MyDrive/ShifaMind'))

# Data paths
DATA_PATH = BASE_PATH / '01_Raw_Datasets'
UMLS_PATH = DATA_PATH / 'Extracted/umls-2025AA-metathesaurus-full/2025AA/META'
ICD10_PATH = DATA_PATH / 'Extracted/icd10cm-CodesDescriptions-2024'
MIMIC_PATH = DATA_PATH / 'Extracted/mimic-iv-3.1'
MIMIC_NOTES_PATH = DATA_PATH / 'Extracted/mimic-iv-note-2.2/note'

# UMLS files
MRCONSO_PATH = UMLS_PATH / 'MRCONSO.RRF'
MRDEF_PATH = UMLS_PATH / 'MRDEF.RRF'
MRSTY_PATH = UMLS_PATH / 'MRSTY.RRF'

# ICD-10 files
ICD10_CODES_PATH = ICD10_PATH / 'icd10cm-codes-2024.txt'

# Model paths
MODEL_PATH = BASE_PATH / '03_Models'
CHECKPOINT_PATH = MODEL_PATH / 'checkpoints'
KNOWLEDGE_BASE_PATH = MODEL_PATH / 'clinical_knowledge_base.json'

# Results paths
RESULTS_PATH = BASE_PATH / '04_Results'
EXPERIMENT_PATH = RESULTS_PATH / 'experiments'

# Ensure directories exist
for path in [MODEL_PATH, CHECKPOINT_PATH, RESULTS_PATH, EXPERIMENT_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Target diagnoses (ICD-10 codes)
TARGET_CODES = ['J189', 'I5023', 'A419', 'K8000']

# ICD-10 descriptions
ICD_DESCRIPTIONS = {
    'J189': 'Pneumonia, unspecified organism',
    'I5023': 'Acute on chronic systolic heart failure',
    'A419': 'Sepsis, unspecified organism',
    'K8000': 'Calculus of gallbladder with acute cholecystitis'
}

# Base model configuration
BASE_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
FUSION_LAYERS = [9, 11]  # Transformer layers to fuse with medical concepts
NUM_ATTENTION_HEADS = 8
HIDDEN_SIZE = 768
DROPOUT = 0.1

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Data configuration
MAX_SEQUENCE_LENGTH = 384
MAX_SAMPLES_PER_DIAGNOSIS = 3000
DEMO_MODE_SAMPLES = 1000

# Training parameters
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS_STAGE1 = 3  # Diagnosis head training
NUM_EPOCHS_STAGE2 = 2  # Concept head training
NUM_EPOCHS_STAGE3 = 3  # Joint fine-tuning
WARMUP_RATIO = 0.1
GRADIENT_CLIP_NORM = 1.0

# Train/val/test split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# CONCEPT CONFIGURATION
# ============================================================================

# Top-N concepts per diagnosis
TOP_N_CONCEPTS_PER_DIAGNOSIS = 15

# Concept filtering thresholds
CONCEPT_SCORE_THRESHOLD = 0.5
MAX_CONCEPTS_OUTPUT = 10
MIN_CONCEPTS_FOR_CITATION = 3

# UMLS sources to use (trusted medical sources)
TRUSTED_UMLS_SOURCES = [
    'SNOMEDCT_US',  # SNOMED CT (US Edition)
    'ICD10CM',      # ICD-10 Clinical Modification
    'MSH',          # Medical Subject Headings (MeSH)
    'NCI',          # National Cancer Institute Thesaurus
    'MEDLINEPLUS',  # MedlinePlus Health Topics
    'HPO'           # Human Phenotype Ontology
]

# ============================================================================
# MEDICAL TERMS FOR CONCEPT SEARCH
# ============================================================================

REQUIRED_MEDICAL_TERMS = {
    'J189': [
        # Core disease
        'Pneumonia', 'Lung infection', 'Respiratory infection',
        # Primary symptoms
        'Fever', 'Cough', 'Dyspnea', 'Shortness of breath',
        # Physical findings
        'Crackles', 'Rales', 'Rhonchi', 'Decreased breath sounds',
        # Labs/vitals
        'Tachypnea', 'Hypoxia', 'Leukocytosis', 'Elevated white blood cell',
        # Imaging
        'Pulmonary infiltrate', 'Lung consolidation',
        # Complications
        'Respiratory distress', 'Hypoxemia', 'Sputum production'
    ],
    'I5023': [
        # Core disease
        'Heart failure', 'Cardiac failure', 'Congestive heart failure', 'Cardiomyopathy',
        # Primary symptoms
        'Dyspnea', 'Shortness of breath', 'Orthopnea', 'Paroxysmal nocturnal dyspnea',
        # Physical findings
        'Edema', 'Swelling', 'Jugular venous distension', 'Pulmonary edema',
        # Cardiac signs
        'S3 gallop', 'Cardiomegaly', 'Pleural effusion',
        # Labs
        'Elevated BNP', 'B-type natriuretic peptide',
        # Other
        'Fatigue', 'Weakness', 'Pulmonary congestion'
    ],
    'A419': [
        # Core disease
        'Sepsis', 'Septicemia', 'Bacteremia', 'Systemic infection',
        # SIRS criteria
        'Fever', 'Hypothermia', 'Tachycardia', 'Tachypnea',
        # Hemodynamics
        'Hypotension', 'Shock', 'Septic shock',
        # Mental status
        'Confusion', 'Altered mental status', 'Delirium',
        # Labs
        'Leukocytosis', 'Leukopenia', 'Lactic acidosis', 'Elevated lactate',
        # Organ dysfunction
        'Organ failure', 'Multi-organ dysfunction', 'Acute kidney injury'
    ],
    'K8000': [
        # Core disease
        'Cholecystitis', 'Gallbladder inflammation', 'Acute cholecystitis',
        'Gallstones', 'Cholelithiasis',
        # Primary symptom
        'Abdominal pain', 'Right upper quadrant pain', 'Biliary colic',
        # Physical findings
        'Murphy sign', 'Abdominal tenderness',
        # Associated symptoms
        'Fever', 'Nausea', 'Vomiting',
        # Labs
        'Leukocytosis', 'Elevated white blood cell count',
        # Imaging
        'Gallbladder wall thickening', 'Pericholecystic fluid'
    ]
}

# Keywords for post-processing filter
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

# ============================================================================
# ANIMAL FILTERING CONFIGURATION
# ============================================================================

# Animal keywords to filter out (veterinary/non-human medicine)
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

# Exclusion patterns (regex)
EXCLUSION_PATTERNS = [
    r'\bof cattle\b',
    r'\bof pigs\b',
    r'\bof swine\b',
    r'\bin cattle\b',
    r'\bin pigs\b',
    r'\bin swine\b',
    r'\bANIMAL\b',
    r'\bVETERINARY\b'
]

# Excluded UMLS semantic types (non-human)
EXCLUDED_SEMANTIC_TYPES = {
    'Mammal',
    'Vertebrate',
    'Animal',
    'Bird',
    'Fish',
    'Amphibian',
    'Reptile',
    'Veterinary Medical Device',
    'Animal-Restricted Concept'
}

# Research/non-clinical semantic types to exclude
EXCLUDED_RESEARCH_TYPES = {
    'Research Activity',
    'Laboratory Procedure',
    'Experimental Model of Disease'
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Metrics computation
CALIBRATION_BINS = 10
NUM_REASONING_CHAIN_EXAMPLES = 10

# Evidence extraction
EVIDENCE_TOP_K = 5
EVIDENCE_SPAN_WINDOW = 10

# Knowledge retrieval
KNOWLEDGE_BASE_TOP_K = 3

# ============================================================================
# DEMO CONFIGURATION
# ============================================================================

# Gradio interface settings
DEMO_PORT = 7860
DEMO_SHARE = True
DEMO_SHOW_ERROR = True

# Output formatting
JSON_INDENT = 2

# ============================================================================
# CHECKPOINT NAMES
# ============================================================================

CHECKPOINT_STAGE1 = CHECKPOINT_PATH / 'shifamind_stage1_diagnosis.pt'
CHECKPOINT_STAGE2 = CHECKPOINT_PATH / 'shifamind_stage2_concepts.pt'
CHECKPOINT_FINAL = CHECKPOINT_PATH / 'shifamind_model.pt'  # Final production model

# ============================================================================
# LOSS FUNCTION WEIGHTS
# ============================================================================

# Stage 3 (joint training) loss weights
LOSS_WEIGHT_DIAGNOSIS = 0.50
LOSS_WEIGHT_CONCEPT = 0.25
LOSS_WEIGHT_CONFIDENCE = 0.25

# Stage 2 (concept head) loss weights
CONCEPT_LOSS_WEIGHT = 0.7
CONFIDENCE_LOSS_WEIGHT = 0.3

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_FILE = 'shifamind.log'
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# ============================================================================
# PERFORMANCE METRICS (for reference)
# ============================================================================

# These are the target/achieved metrics for the model
TARGET_METRICS = {
    'macro_f1': 0.8010,
    'auroc': 0.9122,
    'citation_completeness': 1.00,  # 100%
}

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

def get_device():
    """Get the appropriate device (CUDA/CPU)"""
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_paths():
    """Validate that all required paths exist"""
    required_paths = {
        'UMLS MRCONSO': MRCONSO_PATH,
        'ICD-10 Codes': ICD10_CODES_PATH,
        'Base Path': BASE_PATH,
    }

    missing = []
    for name, path in required_paths.items():
        if not path.exists():
            missing.append(f"{name}: {path}")

    return missing

def print_config_summary():
    """Print a summary of the configuration"""
    print("="*80)
    print("SHIFAMIND CONFIGURATION SUMMARY")
    print("="*80)
    print(f"\nPaths:")
    print(f"  Base Path: {BASE_PATH}")
    print(f"  Model Path: {MODEL_PATH}")
    print(f"  Results Path: {RESULTS_PATH}")
    print(f"\nModel:")
    print(f"  Base Model: {BASE_MODEL_NAME}")
    print(f"  Fusion Layers: {FUSION_LAYERS}")
    print(f"  Max Sequence Length: {MAX_SEQUENCE_LENGTH}")
    print(f"\nTraining:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Epochs (Stage 1/2/3): {NUM_EPOCHS_STAGE1}/{NUM_EPOCHS_STAGE2}/{NUM_EPOCHS_STAGE3}")
    print(f"\nTarget Diagnoses:")
    for code, desc in ICD_DESCRIPTIONS.items():
        print(f"  {code}: {desc}")
    print(f"\nFiltering:")
    print(f"  Animal Keywords: {len(ANIMAL_KEYWORDS)} keywords")
    print(f"  Top-N Concepts per Diagnosis: {TOP_N_CONCEPTS_PER_DIAGNOSIS}")
    print("="*80)

if __name__ == '__main__':
    # Test configuration
    print_config_summary()

    # Validate paths
    missing = validate_paths()
    if missing:
        print("\n WARNING: Missing paths:")
        for path in missing:
            print(f"  {path}")
    else:
        print("\n All required paths exist!")
