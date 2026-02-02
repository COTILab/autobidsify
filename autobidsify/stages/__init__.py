"""BIDS Pipeline Stages"""

from autobidsify.stages.ingest import ingest_data
from autobidsify.stages.evidence import build_evidence_bundle
from autobidsify.stages.classification import classify_files
from autobidsify.stages.trio import (
    trio_generate_all,
    generate_dataset_description,
    generate_readme,
    generate_participants,
)

__all__ = [
    "ingest_data",
    "build_evidence_bundle",
    "classify_files",
    "trio_generate_all",
    "generate_dataset_description",
    "generate_readme",
    "generate_participants",
]
