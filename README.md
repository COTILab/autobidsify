# auto-bidsify

Automated BIDS standardization tool powered by LLM-first architecture.

## Features

- **General compatibility**: Handles diverse dataset structures (flat, hierarchical, multi-site)
- **Multi-modal support**: MRI, fNIRS, and mixed modality datasets
- **Intelligent metadata extraction**: Automatic participant demographics from DICOM headers, documents, and filenames
- **Format conversion**: DICOM→NIfTI, CSV→SNIRF, and more
- **Evidence-based reasoning**: Confidence scoring and provenance tracking for all decisions

## Supported Formats

**Input formats:**
- MRI: DICOM, NIfTI (.nii, .nii.gz)
- fNIRS: SNIRF, Homer3 (.nirs), CSV/TSV tables
- Documents: PDF, DOCX, TXT, Markdown, ...

**Output:** BIDS-compliant dataset (v1.10.0)

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/auto-bidsify.git
cd auto-bidsify

# Setup environment
conda create -n bidsify python=3.10
conda activate bidsify
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-key-here"
```

### Basic Usage

```bash
# Full pipeline (one command)
python cli.py full \
  --input /path/to/your/data \
  --output outputs/my_dataset \
  --model gpt-4o \
  --modality mri

# Step-by-step execution
python cli.py ingest --input data.zip --output outputs/run
python cli.py evidence --output outputs/run --modality mri
python cli.py trio --output outputs/run --model gpt-4o
python cli.py plan --output outputs/run --model gpt-4o
python cli.py execute --output outputs/run
python cli.py validate --output outputs/run
```

### Command Options

```bash
--input PATH          Input data (archive or directory)
--output PATH         Output directory
--model MODEL         LLM model (default: gpt-4o)
--modality TYPE       Data modality: mri|nirs|mixed
--nsubjects N         Number of subjects (optional)
--describe "TEXT"     Dataset description (recommended)
```

## Pipeline Stages

| Stage |   Command   |      Input      |           Output           |               Purpose              |
|-------|-------------|-----------------|----------------------------|------------------------------------|
|   1   | `ingest`    | Raw data        | `ingest_info.json`         | Extract/reference data	          |
|   2   | `evidence`  | All files       | `evidence_bundle.json`     | Analyze structure, detect subjects |
|   3   | `classify`  | Mixed data      | `classification_plan.json` | Separate MRI/fNIRS (optional)      |
|   4   | `trio`      | Evidence   	| BIDS trio files            | Generate metadata files            |
|   5   | `plan`      | Evidence + trio | `BIDSPlan.yaml`            | Create conversion strategy 	  |
|   6   | `execute`   | Plan            | `bids_compatible/`         | Execute conversions 		  |
|   7   | `validate`  | BIDS dataset    | Validation report          | Check compliance 		  |

## Output Structure

```
outputs/my_dataset/
  bids_compatible/              # Final BIDS dataset
    dataset_description.json
    README.md
    participants.tsv
    sub-001/
      anat/
        sub-001_T1w.nii.gz
      func/
        sub-001_task-rest_bold.nii.gz
  _staging/                     # Intermediate files
    evidence_bundle.json
    BIDSPlan.yaml
    conversion_log.json
```

## Examples

### Example 1: Single-site MRI study
```bash
python cli.py full \
  --input brain_scans/ \
  --output outputs/study1 \
  --nsubjects 50 \
  --model gpt-4o \
  --modality mri
```

### Example 2: Multi-site dataset with description
```bash
python cli.py full \
  --input camcan_data/ \
  --output outputs/camcan \
  --model gpt-4o \
  --modality mri \
  --describe "Cambridge Centre for Ageing and Neuroscience: 650 participants, ages 18-88, multi-site MRI study"
```

### Example 3: fNIRS dataset from CSV
```bash
python cli.py full \
  --input fnirs_study/ \
  --output outputs/fnirs \
  --model gpt-4o \
  --modality nirs \
  --describe "Prefrontal cortex activation during cognitive tasks, 30 subjects"
```

## Architecture

**LLM-First Design:**
- **Python**: Deterministic operations (file I/O, format conversion, validation)
- **LLM**: Semantic understanding (file classification, metadata extraction, pattern recognition)
- **Hybrid**: Best of both worlds - reliability + flexibility

## Requirements

- Python 3.10+
- OpenAI API key
- Optional: `dcm2niix` for DICOM conversion
- Optional: `bids-validator` for validation

## Current Status

**Version:** 1.0 (LLM-First Architecture with Evidence-Based Reasoning)

**Tested datasets:**
- Visible Human Project (flat structure, CT scans)
- CamCAN (hierarchical, multi-site, 1288 subjects)
- [Your dataset here - help us test!]

**Known limitations:**
- Classification stage (Stage 3) and mat/spreadsheet conversion is experimental
- Some edge cases in participant metadata extraction

## Contributing

We need YOUR datasets to improve robustness! Please test and report:
- Success cases
- Failure cases  
- Edge cases

