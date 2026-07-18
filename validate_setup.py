#!/usr/bin/env python3
"""
Validate FIDES environment setup.
Checks that all dependencies are installed and accessible.
"""

import sys
from pathlib import Path

print("\n" + "="*80)
print("FIDES ENVIRONMENT VALIDATION")
print("="*80 + "\n")

errors = []
warnings = []

# Step 1: Check Python version
print("1. Checking Python version...")
if sys.version_info < (3, 9):
    errors.append(f"Python 3.9+ required, got {sys.version_info.major}.{sys.version_info.minor}")
else:
    print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# Step 2: Check required packages
print("\n2. Checking required packages...")
required_packages = {
    'pandas': 'Data manipulation',
    'numpy': 'Numerical computing',
    'scipy': 'Scientific computing',
    'sklearn': 'Machine learning (scikit-learn)',
    'statsmodels': 'Statistical modeling',
    'networkx': 'Graph/causal structures',
    'matplotlib': 'Visualization',
    'seaborn': 'Statistical visualization',
}

missing_packages = []
for package, description in required_packages.items():
    try:
        __import__(package)
        print(f"   ✓ {package:20s} {description}")
    except ImportError:
        missing_packages.append(package)
        print(f"   ✗ {package:20s} {description}")

if missing_packages:
    errors.append(f"Missing packages: {', '.join(missing_packages)}")

# Step 3: Check optional packages
print("\n3. Checking optional packages (for GPU/FM inference)...")
optional_packages = {
    'vllm': 'Local LLM inference (needed for real Meditron)',
    'torch': 'PyTorch (needed for FM)',
    'transformers': 'HuggingFace Transformers (needed for FM)',
}

for package, description in optional_packages.items():
    try:
        __import__(package)
        print(f"   ✓ {package:20s} {description}")
    except ImportError:
        print(f"   ⚠ {package:20s} {description}")
        warnings.append(f"{package} not installed - FM inference will use mock")

# Step 4: Check directory structure
print("\n4. Checking directory structure...")
required_dirs = {
    'src/fides': 'FIDES implementation',
    'ablation_study': 'Ablation study code',
    'experiments': 'Causal discovery pipeline',
    'results': 'Output directory (for results)',
}

for dir_path, description in required_dirs.items():
    if Path(dir_path).exists():
        print(f"   ✓ {dir_path:30s} {description}")
    else:
        errors.append(f"Missing directory: {dir_path}")
        print(f"   ✗ {dir_path:30s} {description}")

# Step 5: Check for data directories
print("\n5. Checking for data directories...")
data_dir = Path('results/disease_cohorts')
if data_dir.exists():
    csv_files = list(data_dir.glob('*.csv'))
    if csv_files:
        print(f"   ✓ Found {len(csv_files)} MIMIC data files")
    else:
        warnings.append("No MIMIC data files found - you need to load them")
        print("   ⚠ results/disease_cohorts/ exists but is empty")
        print("     → Load your MIMIC-IV data here before running pipelines")
else:
    print(f"   ⚠ {data_dir} not found")
    print("     → Will be created when you load MIMIC data")

# Step 6: Check imports
print("\n6. Checking FIDES imports...")
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from src.fides import certification, causal, representational
    print("   ✓ FIDES modules importable")
except ImportError as e:
    errors.append(f"Failed to import FIDES modules: {e}")
    print(f"   ✗ {e}")

# Step 7: Check for GPU (optional)
print("\n7. Checking for GPU (optional, for real Meditron)...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"     CUDA version: {torch.version.cuda}")
    else:
        warnings.append("No GPU detected - will use mock Meditron or CPU (slow)")
        print("   ⚠ No GPU detected")
        print("     → Can use mock Meditron 7B or rent cloud GPU ($2)")
except ImportError:
    print("   ⚠ PyTorch not installed - GPU check skipped")

# Final summary
print("\n" + "="*80)
if errors:
    print("❌ VALIDATION FAILED")
    print("="*80)
    for error in errors:
        print(f"  • {error}")
    print("\nFix these errors before running FIDES pipelines.")
    sys.exit(1)
elif warnings:
    print("⚠️  VALIDATION PASSED (with warnings)")
    print("="*80)
    for warning in warnings:
        print(f"  • {warning}")
    print("\nYou can run FIDES, but some features may be limited.")
    print("See warnings above for details.")
    sys.exit(0)
else:
    print("✅ VALIDATION PASSED")
    print("="*80)
    print("\nEnvironment is ready to run FIDES!")
    print("\nNext steps:")
    print("  1. Load MIMIC-IV data into: results/disease_cohorts/")
    print("  2. Read: HANDOFF.md")
    print("  3. Run: python experiments/run_fides_real_causal_discovery.py")
    sys.exit(0)
