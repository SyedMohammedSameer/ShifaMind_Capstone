"""
ShifaMind: Path Configuration Helper

This script helps you configure custom data paths for ShifaMind.
Useful if your data is not at the default Google Drive location.

Usage:
    python setup_paths.py --base-path /path/to/your/data

Author: Mohammed Sameer Syed
Institution: University of Arizona
Date: November 2025
"""

import os
import sys
from pathlib import Path
import argparse


def validate_path_structure(base_path: Path) -> dict:
    """
    Validate that required data files exist at the given base path.

    Args:
        base_path: Base directory containing ShifaMind data

    Returns:
        Dictionary with validation results
    """
    results = {
        'base_exists': base_path.exists(),
        'missing_paths': [],
        'found_paths': []
    }

    # Define required paths relative to base
    required_paths = {
        'Raw Datasets': '01_Raw_Datasets',
        'UMLS MRCONSO': '01_Raw_Datasets/Extracted/umls-2025AA-metathesaurus-full/2025AA/META/MRCONSO.RRF',
        'UMLS MRDEF': '01_Raw_Datasets/Extracted/umls-2025AA-metathesaurus-full/2025AA/META/MRDEF.RRF',
        'UMLS MRSTY': '01_Raw_Datasets/Extracted/umls-2025AA-metathesaurus-full/2025AA/META/MRSTY.RRF',
        'ICD-10 Codes': '01_Raw_Datasets/Extracted/icd10cm-CodesDescriptions-2024/icd10cm-codes-2024.txt',
        'MIMIC Notes': '01_Raw_Datasets/Extracted/mimic-iv-note-2.2/note/discharge.csv.gz',
    }

    for name, rel_path in required_paths.items():
        full_path = base_path / rel_path
        if full_path.exists():
            results['found_paths'].append((name, full_path))
        else:
            results['missing_paths'].append((name, full_path))

    return results


def set_environment_variable(base_path: str):
    """
    Set the SHIFAMIND_BASE_PATH environment variable.

    Args:
        base_path: Path to set as base
    """
    os.environ['SHIFAMIND_BASE_PATH'] = base_path
    print(f"✅ Environment variable set: SHIFAMIND_BASE_PATH={base_path}")
    print("\nNote: This only affects the current Python session.")
    print("For persistent configuration, add this to your notebook:")
    print(f"\n  import os")
    print(f"  os.environ['SHIFAMIND_BASE_PATH'] = '{base_path}'")


def print_validation_report(results: dict, base_path: Path):
    """Print a formatted validation report."""
    print("\n" + "="*70)
    print("PATH VALIDATION REPORT")
    print("="*70)
    print(f"\nBase Path: {base_path}")
    print(f"Base Exists: {'✅ Yes' if results['base_exists'] else '❌ No'}")

    if results['found_paths']:
        print(f"\n✅ Found ({len(results['found_paths'])} items):")
        for name, path in results['found_paths']:
            print(f"   • {name}")

    if results['missing_paths']:
        print(f"\n❌ Missing ({len(results['missing_paths'])} items):")
        for name, path in results['missing_paths']:
            print(f"   • {name}")
            print(f"     Expected at: {path}")

    print("\n" + "="*70)

    if not results['missing_paths']:
        print("✅ All required files found! ShifaMind is ready to run.")
    else:
        print("⚠️  Some files are missing. Please upload them to Google Drive.")
        print("\nTroubleshooting:")
        print("1. Verify files are uploaded to the correct location")
        print("2. Check that folder and file names match exactly")
        print("3. Ensure files are extracted (not still in .zip/.tar.gz)")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Configure ShifaMind data paths',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate default path
  python setup_paths.py

  # Set custom path
  python setup_paths.py --base-path /content/drive/MyDrive/CustomFolder/ShifaMind

  # Validate without setting environment variable
  python setup_paths.py --base-path /path/to/data --no-set
        """
    )

    parser.add_argument(
        '--base-path',
        type=str,
        default='/content/drive/MyDrive/ShifaMind',
        help='Base path to ShifaMind data (default: /content/drive/MyDrive/ShifaMind)'
    )

    parser.add_argument(
        '--no-set',
        action='store_true',
        help='Validate paths only, do not set environment variable'
    )

    args = parser.parse_args()

    # Convert to Path object
    base_path = Path(args.base_path)

    # Validate paths
    print("🔍 Validating ShifaMind data paths...")
    results = validate_path_structure(base_path)

    # Print report
    print_validation_report(results, base_path)

    # Set environment variable if requested
    if not args.no_set and results['base_exists']:
        print()
        set_environment_variable(str(base_path))

    # Return appropriate exit code
    if results['missing_paths']:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
