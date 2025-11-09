# CLI Quick Start Guide

## Complete Pipeline Commands

### 1. Full Pipeline (End-to-End)

```bash
python cli.py full \
    --input data.zip \
    --output bids_out \
    --model gpt-4o \
    --nsubjects 10 \
    --modality mixed \
    --describe "Motor task with fNIRS and fMRI"
```

**Generated files:**
```
bids_out/
├── dataset_description.json
├── README.md
├── participants.tsv
├── _staging/
│   ├── extracted/           # Extracted raw data
│   ├── evidence_bundle.json
│   └── classification.json
└── sub-01/                  # BIDS subject data
    ├── anat/
    │   └── sub-01_T1w.nii.gz
    ├── func/
    │   └── sub-01_task-rest_bold.nii.gz
    └── nirs/
        └── sub-01_task-rest_nirs.snirf
```

---

## Step-by-Step Commands

### Step 1: Ingest Data

```bash
python cli.py ingest \
    --input raw_data.zip \
    --output bids_out
```

**Generated files:**
```
bids_out/
└── _staging/
    └── extracted/
        ├── file1.csv
        ├── file2.nii.gz
        └── protocol.pdf
```

---

### Step 2: Build Evidence Bundle

```bash
python cli.py evidence \
    --output bids_out \
    --nsubjects 10 \
    --modality mixed \
    --describe "Motor task experiment"
```

**Generated files:**
```
bids_out/
└── _staging/
    ├── extracted/           # (preserved from step 1)
    └── evidence_bundle.json # NEW
```

**evidence_bundle.json contains:**
- File structure summary
- Data samples (first few rows of tables, image headers)
- Document content (PDF, DOCX, TXT)
- User hints (nsubjects, modality, description)

---

### Step 3: Classify Files

```bash
python cli.py classify \
    --output bids_out \
    --model gpt-4o
```

**Generated files:**
```
bids_out/
└── _staging/
    ├── extracted/
    ├── evidence_bundle.json
    └── classification.json  # NEW
```

**classification.json contains:**
- `nirs_files`: List of fNIRS data files
- `mri_files`: List of MRI data files
- `unknown_files`: Files that couldn't be classified
- Classification rationale and confidence

---

### Step 4: Generate Trio Files

#### Option A: Generate All Trio Files

```bash
python cli.py trio \
    --output bids_out \
    --model gpt-4o
```

**Generated files:**
```
bids_out/
├── dataset_description.json # NEW
├── README.md                # NEW
├── participants.tsv         # NEW
└── _staging/
    └── ...
```

#### Option B: Generate Individual Files

**Generate dataset_description.json only:**
```bash
python cli.py trio \
    --output bids_out \
    --model gpt-4o \
    --file dataset_description
```

**Generated file:**
```
bids_out/
└── dataset_description.json # NEW
```

**Generate README.md only:**
```bash
python cli.py trio \
    --output bids_out \
    --model gpt-4o \
    --file readme
```

**Generated file:**
```
bids_out/
└── README.md # NEW
```

**Generate participants.tsv only:**
```bash
python cli.py trio \
    --output bids_out \
    --model gpt-4o \
    --file participants
```

**Generated file:**
```
bids_out/
└── participants.tsv # NEW
```

---

### Step 5: NIRS Semantic Discovery (Draft)

```bash
python cli.py nirs-draft \
    --output bids_out \
    --model gpt-4o
```

**Generated files:**
```
bids_out/
└── _staging/
    └── nirs_draft.json # NEW
```

**nirs_draft.json contains:**
- Time column identification
- Channel mappings (source, detector, wavelength)
- Sampling frequency
- Draft SNIRF structure

---

### Step 6: NIRS Semantic Discovery (Normalize)

```bash
python cli.py nirs-normalize \
    --output bids_out \
    --model gpt-4o
```

**Generated files:**
```
bids_out/
└── _staging/
    ├── nirs_draft.json
    └── nirs_normalized.json # NEW
```

**nirs_normalized.json contains:**
- Standardized SNIRF-compliant structure
- Global parameters (SamplingFrequency, Wavelengths)
- File-specific mappings
- Ready for conversion

---

### Step 7: MRI Semantic Discovery (Draft)

```bash
python cli.py mri-draft \
    --output bids_out \
    --model gpt-4o
```

**Generated files:**
```
bids_out/
└── _staging/
    └── mri_draft.json # NEW
```

**mri_draft.json contains:**
- Volume candidates (3D/4D arrays)
- Metadata candidates (voxel size, TR, TE)
- Coordinate system inference
- Confidence scores

---

### Step 8: MRI Semantic Discovery (Finalize)

```bash
python cli.py mri-finalize \
    --output bids_out \
    --model gpt-4o
```

**Generated files:**
```
bids_out/
└── _staging/
    ├── mri_draft.json
    └── mri_final.json # NEW
```

**mri_final.json contains:**
- Executable conversion operations
- Affine transformations
- NIfTI header specifications
- Sidecar JSON metadata

---

### Step 9: Generate BIDS Plan

```bash
python cli.py plan \
    --output bids_out \
    --model gpt-4o
```

**Generated files:**
```
bids_out/
└── _staging/
    └── bids_plan.yaml # NEW
```

**bids_plan.yaml contains:**
- Subject assignment rules
- File mapping rules
- Conversion specifications
- Default metadata values
- Execution fingerprint

---

### Step 10: Execute Conversions

```bash
python cli.py execute \
    --output bids_out
```

**Generated files:**
```
bids_out/
├── dataset_description.json
├── README.md
├── participants.tsv
├── sub-01/
│   ├── anat/
│   │   ├── sub-01_T1w.nii.gz
│   │   └── sub-01_T1w.json
│   ├── func/
│   │   ├── sub-01_task-rest_bold.nii.gz
│   │   └── sub-01_task-rest_bold.json
│   └── nirs/
│       ├── sub-01_task-rest_nirs.snirf
│       ├── sub-01_task-rest_channels.tsv
│       └── sub-01_task-rest_optodes.tsv
├── sub-02/
│   └── ...
└── _staging/
    └── conversion_log.json # NEW
```

**conversion_log.json contains:**
- Conversion success/failure status
- Processed files list
- Error messages (if any)
- Execution timestamp

---

### Step 11: Validate BIDS

```bash
python cli.py validate \
    --output bids_out
```

**Generated files:**
```
bids_out/
└── _staging/
    └── validation_report.txt # NEW
```

**validation_report.txt contains:**
- BIDS validator results
- SNIRF validator results (if applicable)
- Errors and warnings
- Compliance summary

---

## Parameter Reference

### Common Parameters

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `--input` | Input data (ZIP/directory) | Yes (for ingest/full) | - |
| `--output` | Output directory | Yes | - |
| `--model` | LLM model name | Yes (for LLM steps) | `gpt-4o` |
| `--nsubjects` | Number of subjects | No | `1` |
| `--modality` | Data modality hint | No | - |
| `--describe` | Experiment description | No | - |

### Modality Options

- `mri` - MRI data only
- `nirs` - fNIRS data only
- `mixed` - Both MRI and fNIRS

### Model Options

- `gpt-4o` - Default, recommended
- `gpt-4o-mini` - Faster, cheaper
- `o1-preview` - Complex reasoning
- `o1-mini` - Balanced reasoning
- `gpt-5-*` - Future reasoning models

---

## Execution Order

**Full pipeline order:**
```
1. ingest
2. evidence
3. classify
4. trio
5. nirs-draft (if NIRS data)
6. nirs-normalize (if NIRS data)
7. mri-draft (if MRI data)
8. mri-finalize (if MRI data)
9. plan
10. execute
11. validate
```

**You can run steps individually or use `full` command for end-to-end execution.**

---

## Example Workflows

### Workflow 1: MRI Only

```bash
# Full pipeline
python cli.py full \
    --input mri_data.zip \
    --output bids_mri \
    --model gpt-4o \
    --nsubjects 5 \
    --modality mri

# Or step-by-step
python cli.py ingest --input mri_data.zip --output bids_mri
python cli.py evidence --output bids_mri --nsubjects 5 --modality mri
python cli.py classify --output bids_mri --model gpt-4o
python cli.py trio --output bids_mri --model gpt-4o
python cli.py mri-draft --output bids_mri --model gpt-4o
python cli.py mri-finalize --output bids_mri --model gpt-4o
python cli.py plan --output bids_mri --model gpt-4o
python cli.py execute --output bids_mri
python cli.py validate --output bids_mri
```

### Workflow 2: fNIRS Only

```bash
# Full pipeline
python cli.py full \
    --input nirs_data.zip \
    --output bids_nirs \
    --model gpt-4o \
    --nsubjects 3 \
    --modality nirs

# Or step-by-step
python cli.py ingest --input nirs_data.zip --output bids_nirs
python cli.py evidence --output bids_nirs --nsubjects 3 --modality nirs
python cli.py classify --output bids_nirs --model gpt-4o
python cli.py trio --output bids_nirs --model gpt-4o
python cli.py nirs-draft --output bids_nirs --model gpt-4o
python cli.py nirs-normalize --output bids_nirs --model gpt-4o
python cli.py plan --output bids_nirs --model gpt-4o
python cli.py execute --output bids_nirs
python cli.py validate --output bids_nirs
```

### Workflow 3: Regenerate Trio Files Only

```bash
# Regenerate all trio files
python cli.py trio --output existing_bids --model gpt-4o

# Or regenerate specific file
python cli.py trio --output existing_bids --model gpt-4o --file dataset_description
python cli.py trio --output existing_bids --model gpt-4o --file readme
python cli.py trio --output existing_bids --model gpt-4o --file participants
```

---

## Environment Setup

**Before running any commands:**

```bash
export OPENAI_API_KEY="sk-..."
```

**Check if key is set:**

```bash
echo $OPENAI_API_KEY
```
